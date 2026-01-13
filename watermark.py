import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import math

class LogitsGuidedWatermarker:
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
