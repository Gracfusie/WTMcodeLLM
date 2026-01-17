import torch
import torch.nn.functional as F
import hashlib
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

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

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, V]
        Returns:
            entropy: [B]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def _get_green_list_mask(self, proxy_logits: torch.Tensor, last_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proxy_logits: [B, V]
            last_token_ids: [B]
        Returns:
            green_mask: [B, V]
        """
        B, vocab_size = proxy_logits.shape
        device = proxy_logits.device
        
        sorted_indices = torch.argsort(proxy_logits, descending=True, dim=-1)  # [B, V]
        green_mask = torch.zeros(B, vocab_size, dtype=torch.bool, device=device)
        
        def to_signed(val):
            if val >= (1 << 63):
                val -= (1 << 64)
            return val

        SEED_CONST = torch.tensor(to_signed(0x9e3779b97f4a7c15), dtype=torch.int64, device=device)
        KEY_CONST = torch.tensor(to_signed(0x517cc1b727220a95), dtype=torch.int64, device=device)
        MIX_CONST_1 = torch.tensor(to_signed(0xbf58476d1ce4e5b9), dtype=torch.int64, device=device)
        MIX_CONST_2 = torch.tensor(to_signed(0x94d049bb133111eb), dtype=torch.int64, device=device)

        t_last = last_token_ids.to(dtype=torch.int64, device=device)  # [B]
        t_key = torch.tensor(to_signed(int(self.secret_key)), dtype=torch.int64, device=device)

        even_vocab_size = vocab_size - (vocab_size % 2)
        
        if even_vocab_size > 0:
            num_pairs = even_vocab_size // 2
            step_indices = torch.arange(0, even_vocab_size, 2, device=device, dtype=torch.int64)  # [P]
            
            # [B, P] = [1, P] + [B, 1] * scalar + scalar
            h = step_indices.unsqueeze(0) + (t_last.unsqueeze(1) * SEED_CONST) + (t_key * KEY_CONST)
            
            h = (h ^ (h >> 30)) * MIX_CONST_1
            h = (h ^ (h >> 27)) * MIX_CONST_2
            h = h ^ (h >> 31)
            
            bits = (h & 1)
            select_col = (1 - bits).long().unsqueeze(-1)  # [B, P, 1]
            
            pairs = sorted_indices[:, :even_vocab_size].view(B, num_pairs, 2)  # [B, P, 2]
            green_tokens = torch.gather(pairs, 2, select_col).squeeze(-1)  # [B, P]
            
            green_mask.scatter_(1, green_tokens, True)

        if vocab_size % 2 != 0:
            last_token_idx = sorted_indices[:, -1]  # [B]
            t_vocab = torch.tensor(vocab_size, dtype=torch.int64, device=device)
            
            h_last = t_vocab + (t_last * SEED_CONST) + (t_key * KEY_CONST)  # [B]
            
            h_last = (h_last ^ (h_last >> 30)) * MIX_CONST_1
            h_last = (h_last ^ (h_last >> 27)) * MIX_CONST_2
            h_last = h_last ^ (h_last >> 31)
            
            last_is_green = ((h_last & 1) == 1)  # [B]
            batch_indices = torch.arange(B, device=device)[last_is_green]
            if batch_indices.numel() > 0:
                green_mask[batch_indices, last_token_idx[last_is_green]] = True

        return green_mask

    def process_logits(self, last_token_ids: torch.Tensor, main_logits: torch.Tensor, proxy_logits: torch.Tensor, entropy_threshold=None, delta=None):
        """
        Args:
            last_token_ids: [B]
            main_logits: [B, V]
            proxy_logits: [B, V]
        Returns:
            watermarked_logits: [B, V]
            watermarked_mask: [B] bool tensor
        """
        _entropy_threshold = entropy_threshold if entropy_threshold is not None else self.entropy_threshold
        _delta = delta if delta is not None else self.delta
        
        entropy = self._compute_entropy(proxy_logits)  # [B]
        should_watermark = entropy >= _entropy_threshold  # [B]
        
        green_mask = self._get_green_list_mask(proxy_logits, last_token_ids)  # [B, V]
        
        if green_mask.shape[-1] < main_logits.shape[-1]:
            green_mask = F.pad(green_mask, (0, main_logits.shape[-1] - green_mask.shape[-1]))
        
        effective_mask = green_mask & should_watermark.unsqueeze(-1)  # [B, V]
        
        watermarked_logits = torch.where(
            effective_mask, 
            main_logits + _delta, 
            main_logits
        )
        
        return watermarked_logits, should_watermark

class ProxyWatermarkDetector:
    def __init__(self, watermarker, proxy_model, tokenizer, device, 
                 prefix_ids: list[int], suffix_ids: list[int], window_size: int = -1):
        self.watermarker = watermarker
        self.proxy_model = proxy_model
        self.tokenizer = tokenizer #proxy model的tokenizer
        self.device = device
        # proxy model的prefix, suffix以及window
        self.prefix_ids = prefix_ids
        self.suffix_ids = suffix_ids
        self.window_size = window_size

    def detect(self, text: str, z_threshold: float = 4.0):
        #编码 Answer
        text_inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        output_tok_ids = text_inputs.input_ids[0].tolist()
        output_tok_ids = text_inputs.input_ids[0].tolist()
        
        if len(output_tok_ids) == 0:
             return {"error": "Text too short", "prediction": False}

        num_tokens = len(output_tok_ids)
        if len(output_tok_ids) == 0:
             return {"error": "Text too short", "prediction": False}

        num_tokens = len(output_tok_ids)
        green_tokens = 0
        total_scored = 0
        total_scored = 0
        
        #逐个Token检测
        for i in range(num_tokens):
            target_token_id = output_tok_ids[i]
            
            #Prefix + Context(Windowed) + Suffix
            current_generated = output_tok_ids[:i]
            
            if self.window_size > 0 and len(current_generated) > self.window_size:
                context_part = current_generated[-self.window_size:]
            else:
                context_part = current_generated
            
            proxy_in_tok_ids = self.prefix_ids + context_part + self.suffix_ids
            in_tensor = torch.tensor([proxy_in_tok_ids], device=self.device, dtype=torch.long)
            
            #model forward
            target_token_id = output_tok_ids[i]
            
            #Prefix + Context(Windowed) + Suffix
            current_generated = output_tok_ids[:i]
            
            if self.window_size > 0 and len(current_generated) > self.window_size:
                context_part = current_generated[-self.window_size:]
            else:
                context_part = current_generated
            
            proxy_in_tok_ids = self.prefix_ids + context_part + self.suffix_ids
            in_tensor = torch.tensor([proxy_in_tok_ids], device=self.device, dtype=torch.long)
            
            #model forward
            with torch.no_grad():
                outputs = self.proxy_model(input_ids=in_tensor, return_dict=True)
                next_tok_logits = outputs.logits[0, -1, :].unsqueeze(0)
                
            #seed from last token id
            if i > 0:
                last_token_id = output_tok_ids[i-1]
            else:
                last_token_id = 0
                
            #水印判定
            entropy = self.watermarker._compute_entropy(next_tok_logits)
            
            if entropy >= self.watermarker.entropy_threshold:
                total_scored += 1
                
                # 计算红绿名单 mask，形状为 (1, vocab_size)
                green_mask = self.watermarker._get_green_list_mask(next_tok_logits, torch.tensor([last_token_id], device=self.device, dtype=torch.long))
                
                # 判定是否命中
                if target_token_id < green_mask.size(1):
                    is_green = green_mask[0, target_token_id].item()
                    if is_green:
                        green_tokens += 1

        return self._calculate_scores(green_tokens, total_scored, z_threshold)

    def _calculate_scores(self, green_tokens, total_scored, z_threshold):
        if total_scored == 0:
            return {
                "num_green_tokens": 0, "num_tokens_scored": 0, 
                "z_score": 0.0, "p_value": 1.0, "prediction": False, 
                "confidence": 0.0, "green_fraction": 0.0
            }
            
            
        # 没有水印情况下的绿词出现概率：0.5
        gamma = 0.5 
        expected_green = total_scored * gamma
        std_dev = math.sqrt(total_scored * gamma * (1 - gamma))
        
        
        z_score = (green_tokens - expected_green) / std_dev
        prediction = z_score > z_threshold
        green_fraction = green_tokens / total_scored
        
        
        return {
            "num_green_tokens": green_tokens,
            "num_tokens_scored": total_scored,
            "green_fraction": green_fraction,
            "z_score": z_score,
            "prediction": prediction,
            "p_value": 0.0 #先不算，需要吗？
        }