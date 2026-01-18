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
        self.tokenizer = tokenizer
        self.device = device
        self.prefix_ids = prefix_ids
        self.suffix_ids = suffix_ids
        self.window_size = window_size

    def detect(self, text: str | list[str], z_threshold: float = 4.0) -> dict | list[dict]:
        single = isinstance(text, str)
        texts = [text] if single else text
        
        all_tok_ids = [self.tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist() for t in texts]
        max_len = max(len(t) for t in all_tok_ids) if all_tok_ids else 0
        
        if max_len == 0:
            results = [{"num_green_tokens": 0, "num_tokens_scored": 0, "green_fraction": 0.0, "z_score": 0.0, "prediction": False, "p_value": 0.0} for _ in texts]
            return results[0] if single else results
        
        green_counts = [0] * len(texts)
        scored_counts = [0] * len(texts)
        
        for pos in range(max_len):
            batch_indices = [i for i, toks in enumerate(all_tok_ids) if pos < len(toks)]
            if not batch_indices:
                continue
            
            input_seqs = []
            target_tokens = []
            last_tokens = []
            
            for i in batch_indices:
                toks = all_tok_ids[i]
                target_tokens.append(toks[pos])
                last_tokens.append(toks[pos - 1] if pos > 0 else 0)
                context = toks[:pos]
                if self.window_size > 0 and len(context) > self.window_size:
                    context = context[-self.window_size:]
                input_seqs.append(self.prefix_ids + context + self.suffix_ids)
            
            max_input_len = max(len(s) for s in input_seqs)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            padded = [s + [pad_id] * (max_input_len - len(s)) for s in input_seqs]
            attn_masks = [[1] * len(s) + [0] * (max_input_len - len(s)) for s in input_seqs]
            
            in_tensor = torch.tensor(padded, device=self.device, dtype=torch.long)
            attn_tensor = torch.tensor(attn_masks, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                outputs = self.proxy_model(input_ids=in_tensor, attention_mask=attn_tensor, return_dict=True)
            
            seq_lens = [len(s) for s in input_seqs]
            logits_list = [outputs.logits[j, seq_lens[j] - 1, :] for j in range(len(batch_indices))]
            next_tok_logits = torch.stack(logits_list, dim=0)
            
            entropy = self.watermarker._compute_entropy(next_tok_logits)
            last_token_tensor = torch.tensor(last_tokens, device=self.device, dtype=torch.long)
            green_mask = self.watermarker._get_green_list_mask(next_tok_logits, last_token_tensor)
            
            for j, i in enumerate(batch_indices):
                if entropy[j] >= self.watermarker.entropy_threshold:
                    scored_counts[i] += 1
                    if target_tokens[j] < green_mask.size(1) and green_mask[j, target_tokens[j]].item():
                        green_counts[i] += 1
        
        results = []
        for i in range(len(texts)):
            g, t = green_counts[i], scored_counts[i]
            if t == 0:
                results.append({"num_green_tokens": 0, "num_tokens_scored": 0, "green_fraction": 0.0, "z_score": 0.0, "prediction": False, "p_value": 0.0})
            else:
                z = (g - t * 0.5) / math.sqrt(t * 0.25)
                results.append({"num_green_tokens": g, "num_tokens_scored": t, "green_fraction": g / t, "z_score": z, "prediction": z > z_threshold, "p_value": 0.0})
        
        return results[0] if single else results