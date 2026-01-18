from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams

from logits_processors.hf_adapters.toy_debug_processor import (
    ToyDebugLogitsProcessor,
)
from logits_processors.vllm_adapters.hf_logits_processor_adapter import (
    HFLogitsProcessorAdapter,
)
from vllm.multimodal.registry import cached_tokenizer_from_config


class ToyDebugVLLMAdapter(HFLogitsProcessorAdapter):
    """固定参数的玩具调试适配器
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            is_pin_memory=is_pin_memory,
            hf_processor_cls=ToyDebugLogitsProcessor,
            hf_init_kwargs={
                "tokenizer": cached_tokenizer_from_config(
                    vllm_config.model_config
                ),
                "print_first_n": 1,
                "name": "ToyDebug",
            },
            argmax_invariant=False,
            extra_kwargs_from_params=None,
        )

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        # 默认无额外校验
        return None


