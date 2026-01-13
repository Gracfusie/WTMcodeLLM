from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM  # 新增：用于加载 Proxy 模型

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.multimodal.registry import cached_tokenizer_from_config

# 确保 third_party/WLLM 目录在 sys.path (保留原有路径逻辑以防万一)
_WLLM_DIR = Path(__file__).resolve().parents[2] / "third_party" / "WLLM"
if str(_WLLM_DIR) not in sys.path:
    sys.path.insert(0, str(_WLLM_DIR))

# from third_party.WLLM.extended_watermark_processor import WatermarkLogitsProcessor
from logits_processors.vllm_adapters.hf_logits_processor_adapter import (
    HFLogitsProcessorAdapter,
)


class ProxyLogitsGuidedWatermarker:
    """
    1. Entropy-based Switch: Do NOT add watermark when entropy is low
    2. Logits-guided Sampling: 利用小模型的Logits 排序进行红绿名单分组
    """
    def __init__(self, entropy_threshold=0.5, delta=2.0, vocab_size=100, secret_key: int = 42):
        self.entropy_threshold = entropy_threshold
        self.delta = delta
        self.vocab_size = vocab_size
        self.secret_key = secret_key 

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.item()

    def _pseudo_random_hash(self, input_ids: torch.Tensor, step_index: int) -> int:
        # 用key计算hash
        seed_val = input_ids.sum().item() if input_ids is not None else 0
        return (seed_val + step_index + self.secret_key) % 2

    def _get_green_list_mask(self, proxy_logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # 排序Logits
        sorted_indices = torch.argsort(proxy_logits, descending=True, dim=-1)
        
        green_mask = torch.zeros_like(proxy_logits, dtype=torch.bool)
        vocab_size = proxy_logits.size(-1)

        # pair-wise partitioning
        for i in range(0, vocab_size - 1, 2):
            token_a_idx = sorted_indices[0, i]     # 排名第 i 的词
            token_b_idx = sorted_indices[0, i+1]   # 排名第 i+1 的词
            
            # Hash to decide green list
            bit = self._pseudo_random_hash(input_ids, step_index=i)
            
            if bit == 1:
                green_mask[0, token_a_idx] = True # a绿了
            else:
                green_mask[0, token_b_idx] = True # b绿了
                
        #(如果词表是奇数的小补丁)
        if vocab_size % 2 != 0:
            last_token_idx = sorted_indices[0, -1]
            bit = self._pseudo_random_hash(input_ids, step_index=vocab_size)
            if bit == 1:
                green_mask[0, last_token_idx] = True

        return green_mask

    def process_logits(self, input_ids: torch.Tensor, main_logits: torch.Tensor, proxy_logits: torch.Tensor):
        entropy = self._compute_entropy(proxy_logits)
        
        if entropy < self.entropy_threshold:
            return main_logits, False #跳过水印when entropy is low

        #Split
        green_mask = self._get_green_list_mask(proxy_logits, input_ids)
        
        #Insert Bias
        watermarked_logits = main_logits.clone()
        watermarked_logits[green_mask] += self.delta
        
        return watermarked_logits, True


class WatermarkVLLMAdapter(HFLogitsProcessorAdapter):
    """
    vLLM 适配器：集成 ProxyLogitsGuidedWatermarker。
    参数：
    - proxy_model_name: 我们用来生成logits的小模型的名称
    - entropy_threshold: 熵阈值
    - secret_key: 水印密钥
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
        # [CHANGE THIS] proxy logit generator name here
        proxy_model_name: str = "code_completion_model",
        entropy_threshold: float = 0.5,
        secret_key: int = 12345
    ) -> None:
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if tokenizer is None:
            raise RuntimeError("WatermarkVLLMAdapter requires tokenizer from vLLM cache.")

        # initialize class
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            is_pin_memory=is_pin_memory,
            hf_processor_cls=None, 
            hf_init_kwargs={},
            argmax_invariant=False,
            extra_kwargs_from_params=None,
        )

        self.device = device
        
        # Load the proxy model
        print(f"[Watermark] Loading Proxy Model: {proxy_model_name}...")
        self.proxy_model = AutoModelForCausalLM.from_pretrained(proxy_model_name).to(self.device)
        self.proxy_model.eval()

        # initialize watermarking logic
        self.watermarker = ProxyLogitsGuidedWatermarker(
            entropy_threshold=entropy_threshold,
            delta=2.0,
            vocab_size=len(tokenizer.get_vocab()),
            secret_key=secret_key
        )

    def __call__(self, prompt_tokens_ids, past_token_ids, scores):
        # 准备 Input IDs
        current_seq_ids = prompt_tokens_ids + past_token_ids
        input_ids = torch.tensor([current_seq_ids], device=self.device)

        # 运行 Proxy 模型 (No Grad)
        with torch.no_grad():

            # [(maybe)CHANGE THIS]为了防止OOM这里简单的写了截断逻辑
            proxy_input = input_ids if input_ids.shape[1] < 1024 else input_ids[:, -1024:]
            proxy_outputs = self.proxy_model(proxy_input)
            proxy_logits = proxy_outputs.logits[:, -1, :]

            # [(maybe)CHANGE THIS] 把词表粗爆对齐
            if proxy_logits.shape[-1] != scores.shape[-1]:
                min_vocab = min(proxy_logits.shape[-1], scores.shape[-1])
                proxy_logits = proxy_logits[:, :min_vocab]

        # 调用水印逻辑
        final_logits, applied = self.watermarker.process_logits(
            input_ids=input_ids,
            main_logits=scores,
            proxy_logits=proxy_logits
        )

        return final_logits

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        return None

