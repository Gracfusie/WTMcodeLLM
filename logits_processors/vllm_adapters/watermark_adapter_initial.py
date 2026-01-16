from __future__ import annotations

import sys
from pathlib import Path

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.multimodal.registry import cached_tokenizer_from_config

# 确保 third_party/WLLM 目录在 sys.path，解决内部相对 import（normalizers 等）
_WLLM_DIR = Path(__file__).resolve().parents[2] / "third_party" / "WLLM"
if str(_WLLM_DIR) not in sys.path:
    sys.path.insert(0, str(_WLLM_DIR))

from third_party.WLLM.extended_watermark_processor import WatermarkLogitsProcessor
from logits_processors.vllm_adapters.hf_logits_processor_adapter_initial import (
    HFLogitsProcessorAdapter,
)


class WatermarkVLLMAdapter(HFLogitsProcessorAdapter):
    """vLLM 侧固定参数的 Watermark 适配器（使用 extended_watermark_processor）。

    - 不复制实现，直接 import `WatermarkLogitsProcessor`。
    - 固定参数：gamma=0.25, delta=2.0, seeding_scheme=\"selfhash\"（与 README 示例保持一致）。
    - vocab 直接取 tokenizer.get_vocab() 的 values。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if tokenizer is None:
            raise RuntimeError("WatermarkVLLMAdapter requires tokenizer from vLLM cache.")

        super().__init__(
            vllm_config=vllm_config,
            device=device,
            is_pin_memory=is_pin_memory,
            hf_processor_cls=WatermarkLogitsProcessor,
            hf_init_kwargs={
                "vocab": list(tokenizer.get_vocab().values()),
                "gamma": 0.25,
                "delta": 2.0,
                "seeding_scheme": "selfhash",
            },
            argmax_invariant=False,
            extra_kwargs_from_params=None,
        )

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        # 无额外校验
        return None
