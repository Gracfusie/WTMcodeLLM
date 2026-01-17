from __future__ import annotations
import os
import time
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.sample.logits_processor import (BatchUpdate,
                                            LogitsProcessor,
                                            MoveDirectionality)

from logits_processors.utils.proxy_model import ProxyModelManager
from logits_processors.utils.watermark import ProxyLogitsGuidedWatermarker
from config import EnvConfig

import sys
from pathlib import Path

# 确保 third_party/WLLM 目录在 sys.path，解决内部相对 import（normalizers 等）
_WLLM_DIR = Path(__file__).resolve().parents[2] / "third_party" / "WLLM"
if str(_WLLM_DIR) not in sys.path:
    sys.path.insert(0, str(_WLLM_DIR))

from third_party.WLLM.extended_watermark_processor import WatermarkLogitsProcessor
from logits_processors.vllm_adapters.hf_logits_processor_adapter_initial import (
    HFLogitsProcessorAdapter,
)

_CODE_WATERMARK_DIR = Path(__file__).resolve().parents[3] / "third_party" / "code_watermark"
if str(_CODE_WATERMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_WATERMARK_DIR))

try:
    from models.sweet import SweetLogitsProcessor
except ImportError:
    SweetLogitsProcessor = None


class WLLM_VLLMAdapter(HFLogitsProcessorAdapter):
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
                "gamma": os.environ.get("WATERMARK_GAMMA"),
                "delta": os.environ.get("WATERMARK_DELTA"),
                "seeding_scheme": "selfhash",
            },
            argmax_invariant=False,
            extra_kwargs_from_params=None,
        )

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        # 无额外校验
        return None


class Proxy_VLLMAdapter(LogitsProcessor):
    """将 HuggingFace 的 LogitsProcessor 适配为 vLLM 批处理接口。支持 Batch Update。"""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                is_pin_memory: bool):
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.proxy_model_manager = ProxyModelManager(main_tokenizer=self.tokenizer, device=device, is_pin_memory=is_pin_memory)
        # 这个 vocab size 真的对吗？感觉不太对，需要再检查一下
        self.watermarker = ProxyLogitsGuidedWatermarker(
            entropy_threshold=EnvConfig.watermark_entropy_threshold, 
            delta=EnvConfig.watermark_delta, 
            vocab_size=len(self.proxy_model_manager.main_tokenizer.get_vocab()), 
            secret_key=EnvConfig.watermark_secret_key
        )
        self.debug_enabled = os.environ.get('WATERMARK_DEBUG', '0') == '1'

    @classmethod
    def validate_params(cls, params: SamplingParams):
        return ProxyModelManager.validate_params(params)

    # 最高概率 token 可能在 redlist
    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if self.debug_enabled:
            start_time = time.perf_counter()
        result = self.proxy_model_manager.update_state(batch_update)
        if self.debug_enabled:
            elapsed = time.perf_counter() - start_time
            print(f"[TIMING] update_state: {elapsed*1000:.2f}ms")
        return result

    def _debug_output(self, input_ids: torch.Tensor, main_logits: torch.Tensor, proxy_logits: torch.Tensor, watermarked_logits: torch.Tensor, applied: bool):
        print("\n" + "="*80)
        print(f"WATERMARK DEBUG - Applied: {applied}")
        print("="*80)
        
        context_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        print(f"Context: {repr(context_text)}")
        
        vocab_size = min(proxy_logits.shape[-1], main_logits.shape[-1])
        main_probs_full = F.softmax(main_logits[0][:vocab_size], dim=-1)
        proxy_probs_full = F.softmax(proxy_logits[0][:vocab_size], dim=-1)
        
        kl_divergence = F.kl_div(
            F.log_softmax(proxy_logits[0][:vocab_size], dim=-1),
            main_probs_full,
            reduction='batchmean'
        ).item()
        
        print(f"\nKL Divergence (proxy||main): {kl_divergence:.6f}")
        
        main_probs = F.softmax(main_logits[0], dim=-1)
        main_top10 = torch.topk(main_probs, k=10)
        print("\nMain Model Top-10:")
        for i, (prob, idx) in enumerate(zip(main_top10.values.tolist(), main_top10.indices.tolist())):
            token_text = self.tokenizer.decode([idx])
            print(f"  {i+1}. [{idx:5d}] {repr(token_text):20s} {prob:.6f}")
        
        proxy_probs = F.softmax(proxy_logits[0], dim=-1)
        proxy_top10 = torch.topk(proxy_probs, k=10)
        print("\nProxy Model Top-10:")
        for i, (prob, idx) in enumerate(zip(proxy_top10.values.tolist(), proxy_top10.indices.tolist())):
            token_text = self.tokenizer.decode([idx])
            print(f"  {i+1}. [{idx:5d}] {repr(token_text):20s} {prob:.6f}")
        
        watermarked_probs = F.softmax(watermarked_logits[0], dim=-1)
        watermarked_top10 = torch.topk(watermarked_probs, k=10)
        print("\nWatermarked Model Top-10:")
        for i, (prob, idx) in enumerate(zip(watermarked_top10.values.tolist(), watermarked_top10.indices.tolist())):
            token_text = self.tokenizer.decode([idx])
            print(f"  {i+1}. [{idx:5d}] {repr(token_text):20s} {prob:.6f}")
        
        print("="*80 + "\n")

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.debug_enabled:
            start_time = time.perf_counter()

        # sanity check
        assert len(logits) == len(self.proxy_model_manager.req_info_by_idx), "logits 和 request 数量不一致"

        for i in range(len(logits)):
            req_state = self.proxy_model_manager.req_info_by_idx[i]

            if req_state.output_tok_ids:
                last_token_id = req_state.output_tok_ids[-1]
            else:
                last_token_id = 0

            main_logits_unsqueezed = logits[i].unsqueeze(0)
            proxy_logits_unsqueezed = req_state.next_tok_logits.unsqueeze(0)
            
            new_logits, applied = self.watermarker.process_logits(
                last_token_id=last_token_id,
                main_logits=main_logits_unsqueezed,
                proxy_logits=proxy_logits_unsqueezed,
                entropy_threshold=req_state.entropy_threshold,
                delta=req_state.delta,
            )
            
            if applied:
                req_state.watermarked_count += 1
            else:
                req_state.non_watermarked_count += 1
            
            if self.debug_enabled:
                input_ids = torch.tensor(req_state.prompt_tok_ids + req_state.output_tok_ids, device=logits.device, dtype=torch.long).unsqueeze(0)
                self._debug_output(input_ids, main_logits_unsqueezed, proxy_logits_unsqueezed, new_logits, applied)
            
            logits[i] = new_logits.squeeze(0)

        if self.debug_enabled:
            elapsed = time.perf_counter() - start_time
            print(f"[TIMING] apply: {elapsed*1000:.2f}ms")

        return logits

class Sweet_VLLMAdapter(HFLogitsProcessorAdapter):
    """
    SWEET 适配器
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        if SweetLogitsProcessor is None:
             raise ImportError("SWEET dependencies not found.")

        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        
        entropy_threshold = float(os.environ.get("WATERMARK_ENTROPY_THRESHOLD", 1.5))
        gamma = float(os.environ.get("WATERMARK_GAMMA", 0.5))
        delta = float(os.environ.get("WATERMARK_DELTA", 2.0))
        secret_key = int(os.environ.get("WATERMARK_SECRET_KEY", 42))

        super().__init__(
            vllm_config=vllm_config,
            device=device,
            is_pin_memory=is_pin_memory,
            hf_processor_cls=SweetLogitsProcessor,
            hf_init_kwargs={
                "vocab": list(tokenizer.get_vocab().values()),
                "gamma": gamma,
                "delta": delta,
                "entropy_threshold": entropy_threshold,
                "hash_key": secret_key,                
                "seeding_scheme": "selfhash"           
            },
            argmax_invariant=False,
            extra_kwargs_from_params=None,
        )

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        return None


class WatermarkVLLMAdapter(LogitsProcessor):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        self.algorithm = os.environ.get("WATERMARK_ALGORITHM", "proxy").lower()
        print(f"[WatermarkAdapter] Loading Strategy: {self.algorithm.upper()}")
        if self.algorithm == "proxy":
            self.impl = Proxy_VLLMAdapter(vllm_config, device, is_pin_memory)
            
        elif self.algorithm == "wllm":
            self.impl = WLLM_VLLMAdapter(vllm_config, device, is_pin_memory)

        elif self.algorithm=="sweet":
            self.impl=Sweet_VLLMAdapter(vllm_config, device, is_pin_memory)
        else:
            raise ValueError(f"Unknown watermark algorithm: {self.algorithm}")

    @classmethod
    def validate_params(cls, params: SamplingParams):
        return Proxy_VLLMAdapter.validate_params(params)

    def is_argmax_invariant(self) -> bool:
        return self.impl.is_argmax_invariant()

    def update_state(self, batch_update: BatchUpdate | None):
        return self.impl.update_state(batch_update)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return self.impl.apply(logits)