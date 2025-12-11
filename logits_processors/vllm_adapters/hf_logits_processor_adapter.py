from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Optional, Type

import torch
from transformers import LogitsProcessor as HFLogitsProcessor
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.multimodal.registry import cached_tokenizer_from_config


class GPTOSSChannelState(Enum):
    """GPT-OSS 对话通道状态。"""

    NO_CHANNEL = "no_channel"
    AWAITING_CHANNEL_NAME = "awaiting_channel_name"
    FINAL = "final"
    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    UNKNOWN_CHANNEL = "unknown_channel"


class HFLogitsProcessorAdapter(AdapterLogitsProcessor):
    """将 HuggingFace 的 request 级 LogitsProcessor 适配为 vLLM 批处理接口。

    适用于拥有 __call__(input_ids, scores) -> scores 签名的 HF LogitsProcessor。
    运行时为每个 request 维护一个独立的 HF processor 实例，使用 prompt_ids +
    output_ids 组装 input_ids 张量，再对 logits 行进行改写。

    vLLM 侧实例通常不再暴露额外参数；HF 侧具体参数通过 hf_init_kwargs /
    extra_kwargs_from_params 固定在实例中。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
        hf_processor_cls: Type[HFLogitsProcessor],
        hf_init_kwargs: Optional[Dict[str, Any]] = None,
        argmax_invariant: bool = False,
        extra_kwargs_from_params: Optional[
            Callable[[SamplingParams], Dict[str, Any]]
        ] = None,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        self.hf_processor_cls = hf_processor_cls
        self.hf_init_kwargs = hf_init_kwargs or {}
        self._argmax_invariant = argmax_invariant
        # 若需要从 SamplingParams 读取自定义参数创建 HF processor，可提供回调
        self.extra_kwargs_from_params = extra_kwargs_from_params
        # GPT-OSS channel 识别（用于 gating，仅 FINAL 走 HF processor）
        self._gptoss_enabled = False
        self._gptoss_special_ids_set: set[int] = set()
        self._gptoss_start_id: Optional[int] = None
        self._gptoss_channel_id: Optional[int] = None
        self._gptoss_final_first_id: Optional[int] = None
        self._gptoss_analysis_first_id: Optional[int] = None
        self._gptoss_commentary_first_id: Optional[int] = None
        self._init_gptoss_detector(vllm_config)
        if not self._gptoss_enabled:
            raise RuntimeError("GPT-OSS channel detection is required but failed to initialize.")

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        # 默认不做额外校验，使用者可按需覆写
        return None

    def is_argmax_invariant(self) -> bool:
        return self._argmax_invariant

    def _init_gptoss_detector(self, vllm_config: VllmConfig) -> None:
        """初始化 GPT-OSS channel 检测；失败即报错，绝不静默降级。"""

        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if tokenizer is None:
            raise RuntimeError("GPT-OSS detection needs tokenizer from config, got None.")

        special_ids_set = set(tokenizer.all_special_ids)
        for tid, tok in tokenizer.added_tokens_decoder.items():
            if getattr(tok, "special", False):
                special_ids_set.add(tid)

        start_id = tokenizer.convert_tokens_to_ids("<|start|>")
        channel_id = tokenizer.convert_tokens_to_ids("<|channel|>")
        if start_id is None or channel_id is None:
            raise RuntimeError("GPT-OSS detection needs <|start|> and <|channel|> tokens.")

        def _first_id(tok: str) -> Optional[int]:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            return ids[0] if ids else None

        self._gptoss_special_ids_set = special_ids_set
        self._gptoss_start_id = start_id
        self._gptoss_channel_id = channel_id
        self._gptoss_final_first_id = _first_id("final")
        self._gptoss_analysis_first_id = _first_id("analysis")
        self._gptoss_commentary_first_id = _first_id("commentary")
        self._gptoss_enabled = True

    def _gptoss_channel_state(self, prefix_ids: list[int]) -> GPTOSSChannelState:
        if not self._gptoss_enabled:
            raise RuntimeError("GPT-OSS channel detection not initialized.")
        if not prefix_ids:
            return GPTOSSChannelState.NO_CHANNEL

        for idx in range(len(prefix_ids) - 1, -1, -1):
            tid = prefix_ids[idx]
            if tid not in self._gptoss_special_ids_set:
                continue

            if tid == self._gptoss_channel_id:
                if idx == len(prefix_ids) - 1:
                    return GPTOSSChannelState.AWAITING_CHANNEL_NAME
                if idx + 1 < len(prefix_ids):
                    next_id = prefix_ids[idx + 1]
                    if self._gptoss_final_first_id is not None and next_id == self._gptoss_final_first_id:
                        return GPTOSSChannelState.FINAL
                    if self._gptoss_analysis_first_id is not None and next_id == self._gptoss_analysis_first_id:
                        return GPTOSSChannelState.ANALYSIS
                    if self._gptoss_commentary_first_id is not None and next_id == self._gptoss_commentary_first_id:
                        return GPTOSSChannelState.COMMENTARY
                    return GPTOSSChannelState.UNKNOWN_CHANNEL
            if tid == self._gptoss_start_id:
                return GPTOSSChannelState.NO_CHANNEL

        return GPTOSSChannelState.NO_CHANNEL

    def _build_hf_processor(self, params: SamplingParams) -> HFLogitsProcessor:
        extra_kwargs = (
            self.extra_kwargs_from_params(params)
            if self.extra_kwargs_from_params
            else {}
        )
        kwargs = {**self.hf_init_kwargs, **extra_kwargs}
        return self.hf_processor_cls(**kwargs)

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ):
        """为单个 request 创建包裹函数，供 AdapterLogitsProcessor 调用。"""
        hf_proc = self._build_hf_processor(params)

        def _req_lp(
            prompt_ids: Optional[list[int]],
            output_ids: list[int],
            logits: torch.Tensor,
        ) -> torch.Tensor:
            # 将 prompt + 已生成输出拼接成 HF 期望的 input_ids（batch=1）
            input_ids_list = (prompt_ids or []) + output_ids

            # GPT-OSS thinking 特殊处理：仅 FINAL 走 HF processor，其余直接透传 logits
            channel_state = self._gptoss_channel_state(input_ids_list)
            # if channel_state is not None:
            #     # 临时调试输出：观测当前通道类型
            #     print(
            #         f"[HFLogitsProcessorAdapter][GPT-OSS] channel_state={channel_state.value} "
            #         f"prompt_len={len(prompt_ids or [])} output_len={len(output_ids)}"
            #     )
            if channel_state is not None and channel_state is not GPTOSSChannelState.FINAL:
                return logits

            input_ids = torch.tensor(
                [input_ids_list], device=logits.device, dtype=torch.long
            )
            scores = logits.unsqueeze(0)
            with torch.no_grad():
                out = hf_proc(input_ids=input_ids, scores=scores)
            # HF processor 可能原地改写 scores，这里取返回值的第一行
            return out[0]

        return _req_lp


