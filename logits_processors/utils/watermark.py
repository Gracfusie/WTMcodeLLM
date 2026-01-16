import torch
import torch.nn.functional as F
import hashlib
import math

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

    def _get_green_list_mask(self, proxy_logits: torch.Tensor, last_token_id: int) -> torch.Tensor:
        sorted_indices = torch.argsort(proxy_logits, descending=True, dim=-1)
        
        vocab_size = proxy_logits.size(-1)
        device = proxy_logits.device
        
        green_mask = torch.zeros_like(proxy_logits, dtype=torch.bool)
        
        # 将无符号的 64位 hex 转为 有符号的 int64
        def to_signed(val):
            if val >= (1 << 63):
                val -= (1 << 64)
            return val

        SEED_CONST = torch.tensor(to_signed(0x9e3779b97f4a7c15), dtype=torch.int64, device=device)
        KEY_CONST = torch.tensor(to_signed(0x517cc1b727220a95), dtype=torch.int64, device=device)
        MIX_CONST_1 = torch.tensor(to_signed(0xbf58476d1ce4e5b9), dtype=torch.int64, device=device)
        MIX_CONST_2 = torch.tensor(to_signed(0x94d049bb133111eb), dtype=torch.int64, device=device)

        t_last = torch.tensor(last_token_id, dtype=torch.int64, device=device)
        t_key = torch.tensor(to_signed(int(self.secret_key)), dtype=torch.int64, device=device)

        even_vocab_size = vocab_size - (vocab_size % 2)
        
        if even_vocab_size > 0:
            step_indices = torch.arange(0, even_vocab_size, 2, device=device, dtype=torch.int64)
            
            h = step_indices + (t_last * SEED_CONST) + (t_key * KEY_CONST)
            
            h = (h ^ (h >> 30)) * MIX_CONST_1
            h = (h ^ (h >> 27)) * MIX_CONST_2
            h = h ^ (h >> 31)
            
            bits = (h & 1) 

            select_col = (1 - bits).long().unsqueeze(1)
            pairs = sorted_indices[0, :even_vocab_size].view(-1, 2)
            green_tokens = torch.gather(pairs, 1, select_col).squeeze(1)
            green_mask[0].scatter_(0, green_tokens, True)

        # Corner case
        if vocab_size % 2 != 0:
            last_token_idx = sorted_indices[0, -1]
            t_vocab = torch.tensor(vocab_size, dtype=torch.int64, device=device)
            
            h_last = t_vocab + (t_last * SEED_CONST) + (t_key * KEY_CONST)
            
            h_last = (h_last ^ (h_last >> 30)) * MIX_CONST_1
            h_last = (h_last ^ (h_last >> 27)) * MIX_CONST_2
            h_last = h_last ^ (h_last >> 31)
            
            if (h_last & 1) == 1:
                green_mask[0, last_token_idx] = True

        return green_mask

    def process_logits(self, last_token_id: int, main_logits: torch.Tensor, proxy_logits: torch.Tensor, entropy_threshold=None, delta=None):
        _entropy_threshold = entropy_threshold if entropy_threshold is not None else self.entropy_threshold
        _delta = delta if delta is not None else self.delta
        
        entropy = self._compute_entropy(proxy_logits)
    
        if entropy < _entropy_threshold:
            return main_logits, False #跳过水印when entropy is low
        #Split
        green_mask = self._get_green_list_mask(proxy_logits, last_token_id)
        if green_mask.shape[-1] < main_logits.shape[-1]:
            # pad 0。尽量不去鼓励非公共 tok
            green_mask = torch.nn.functional.pad(green_mask, (0, main_logits.shape[-1] - green_mask.shape[-1]))
        
        #Insert Bias
        watermarked_logits = main_logits.clone()
        watermarked_logits[green_mask] += _delta
        
        return watermarked_logits, True

class ProxyWatermarkDetector:
    def __init__(self, watermarker, proxy_model, tokenizer, device):
        self.watermarker = watermarker
        self.proxy_model = proxy_model
        self.tokenizer = tokenizer
        self.device = device

    def detect(self, text: str, context: str = None, z_threshold: float = 4.0):
        if context:
            context_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=True)
            context_ids = context_inputs.input_ids.to(self.device)
        else:
            context_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)

        text_inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        text_ids = text_inputs.input_ids.to(self.device)
        
        if text_ids.size(1) == 0:
             return {"error": "Text too short"}

        # 初始化变量
        past_key_values = None
        last_token_id = None
        next_token_logits = None
        # 整个 context 一次性喂给 Proxy
        if context_ids.size(1) > 0:
            with torch.no_grad():
                outputs = self.proxy_model(context_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                last_token_id = context_ids[0, -1].item()

        num_tokens = text_ids.size(1)
        green_tokens = 0
        total_scored = 0 
        
        for i in range(num_tokens):
            target_token_id = text_ids[0, i].item() #current word
            if next_token_logits is not None:
                #watermark logic

                #entropy check
                entropy = self.watermarker._compute_entropy(next_token_logits)
                
                if entropy >= self.watermarker.entropy_threshold:
                    total_scored += 1
                    green_mask = self.watermarker._get_green_list_mask(next_token_logits, last_token_id)
                    
                    if target_token_id < green_mask.size(1):
                        if green_mask[0, target_token_id]:
                            green_tokens += 1
            
            #update KV cache
            current_input = text_ids[:, i].unsqueeze(1) # shape [1, 1]
            
            with torch.no_grad():
                outputs = self.proxy_model(current_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                last_token_id = target_token_id #update seed

        return self._calculate_scores(green_tokens, total_scored, z_threshold)

    def _calculate_scores(self, green_tokens, total_scored, z_threshold):
        if total_scored == 0:
            return {"num_green_tokens": 0, "num_tokens_scored": 0, "z_score": 0.0, "p_value": 1.0, "prediction": False, "confidence": 0.0, "green_fraction": 0.0}
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
            "p_value": 0.0
        }