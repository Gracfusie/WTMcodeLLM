import torch
import torch.nn.functional as F
import math
import scipy.stats

class SweetDetector:
    """
    SweetDetector
    
    - Seeding Scheme: 'simple_1' (Context Width = 1)
    - Hash Logic: rng.manual_seed(hash_key * prev_token)
    - Filtering: 只统计 Entropy > Threshold 的 token
    """
    def __init__(
        self, 
        entropy_threshold: float, 
        hash_key: int, 
        vocab_size: int = None, 
        gamma: float = 0.5, 
        z_threshold: float = 4.0,
        tokenizer = None  
    ):
        self.entropy_threshold = entropy_threshold
        self.hash_key = hash_key
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.z_threshold = z_threshold
        self.tokenizer = tokenizer  # 保存 tokenizer
        
        # 对应 simple_1
        self.min_prefix_len = 1 
        
        # 初始化 RNG
        self.rng = torch.Generator()
        self.device = torch.device("cpu") 
        
        print(f"[SweetDetector] Configured: Threshold={self.entropy_threshold}, Key={self.hash_key}, Gamma={self.gamma}, Scheme=simple_1")

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """
        核心哈希对齐函数
        对应生成代码中的: self.rng.manual_seed(hash_key * prev_token)
        """
        # 确保至少有 1 个 token 作为 context
        if input_ids.shape[-1] < 1:
            raise ValueError(f"seeding_scheme='simple_1' requires at least 1 token prefix.")
            
        prev_token = input_ids[-1].item()
        
        self.rng.manual_seed(self.hash_key * prev_token) 

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        self._seed_rng(input_ids)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        
        return greenlist_ids.tolist()

    def _compute_entropy(self, logits: torch.Tensor) -> list[float]:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Entropy = - sum(p * log p)
        # 处理 0 概率的情况，防止 NaN
        entropies = -torch.sum(torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs)), dim=-1)
        return entropies.tolist()

    def _compute_z_score(self, observed_count, T):
        if T == 0: return 0.0
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = math.sqrt(T * expected_count * (1 - expected_count))
        if denom == 0: return 0.0
        return numer / denom

    def _compute_p_value(self, z):
        return scipy.stats.norm.sf(z)

    def detect_with_model(self, text: str, model, tokenizer=None, gold_prompt: str = ""):
        """
        检测主入口
        """
        self.device = model.device
        
        # 优先使用传入的 tokenizer，没有则用 init 时的
        current_tokenizer = tokenizer if tokenizer else self.tokenizer
        
        if current_tokenizer is None:
             raise ValueError("Tokenizer missing. Please pass tokenizer to __init__ or detect_with_model.")

        if self.vocab_size is None:
            self.vocab_size = len(current_tokenizer.get_vocab())
            
        # 1. 编码输入
        if gold_prompt:
            prompt_inputs = current_tokenizer(gold_prompt, return_tensors="pt", add_special_tokens=False)
            prompt_ids = prompt_inputs["input_ids"][0].to(self.device)
            prompt_len = len(prompt_ids)
        else:
            prompt_ids = torch.tensor([], device=self.device, dtype=torch.long)
            prompt_len = 0

        text_inputs = current_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        text_ids = text_inputs["input_ids"][0].to(self.device)
        
        # 拼接完整序列用于计算熵
        full_input_ids = torch.cat([prompt_ids, text_ids], dim=0)
        
        if len(text_ids) == 0:
            return {"error": "Text empty", "prediction": False}

        with torch.no_grad():
            outputs = model(full_input_ids.unsqueeze(0))
            # Logits 形状: [SeqLen, Vocab]
            all_logits = outputs.logits[0]

        start_idx = max(0, prompt_len - 1)
        
        if prompt_len > 0:
            relevant_logits = all_logits[start_idx : start_idx + len(text_ids)]
            entropy_list = self._compute_entropy(relevant_logits)
        else:
            # 如果没有 Prompt，第一个词无法计算熵（因为它没有上文）
            if len(text_ids) > 1:
                relevant_logits = all_logits[0 : len(text_ids) - 1]
                partial_entropy = self._compute_entropy(relevant_logits)
                # 第一个词补 0.0 (视为低熵，跳过检测)
                entropy_list = [0.0] + partial_entropy
            else:
                entropy_list = [0.0]

        # 截断以确保长度一致
        entropy_list = entropy_list[:len(text_ids)]

        return self._score_sequence(
            input_ids=text_ids,
            entropy=entropy_list
        )

    def detect_without_model(self, text: str | list[str], current_tokenizer=None):
        current_tokenizer = self.tokenizer if current_tokenizer is None else current_tokenizer
        if current_tokenizer is None:
             raise ValueError("Tokenizer missing. Please pass tokenizer to __init__ or detect_with_model.")

        if self.vocab_size is None:
            self.vocab_size = len(current_tokenizer.get_vocab())

        def detect_single_str(t: str):
            text_inputs = current_tokenizer(t, return_tensors="pt", add_special_tokens=False)
            text_ids = text_inputs["input_ids"][0].to(self.device)
            
            return self._score_sequence(
                input_ids=text_ids,
                entropy=[1000. ] * len(text_ids)
            )
        
        if isinstance(text, str):
            return detect_single_str(text)
        elif isinstance(text, list):
            return [detect_single_str(t) for t in text]
        else:
            raise ValueError(f"Invalid text type: {type(text)}")

    def _score_sequence(self, input_ids, entropy):
        # 减去前缀长度 (simple_1 需要 1 个 token 做前缀，所以从第 2 个词开始检测)
        prefix_len = self.min_prefix_len 
        num_tokens_generated = len(input_ids) - prefix_len
        
        if num_tokens_generated < 1:
            return {"invalid": True, "reason": "Text too short"}

        green_token_count = 0
        num_tokens_scored = 0 # 只统计高熵的 token
        
        # 遍历生成的 token (跳过前缀)
        for i in range(prefix_len, len(input_ids)):
            token_entropy = entropy[i]
            
            # --- SWEET 核心逻辑 ---
            # 只有当熵大于阈值时，才进行水印检测
            if token_entropy > self.entropy_threshold:
                num_tokens_scored += 1
                
                curr_token = input_ids[i].item()
                
                # 获取 context: simple_1 只需要前一个 token
                # input_ids[:i] 的最后一个元素就是 input_ids[i-1]
                greenlist_ids = self._get_greenlist_ids(input_ids[:i])
                
                if curr_token in greenlist_ids:
                    green_token_count += 1

        # 统计结果
        if num_tokens_scored == 0:
             return {
                "num_total_tokens": num_tokens_generated,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "green_fraction": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "prediction": False
            }

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        p_value = self._compute_p_value(z_score)
        
        return {
            "num_total_tokens": num_tokens_generated,
            "num_green_tokens": green_token_count,
            "num_tokens_scored": num_tokens_scored,
            "green_fraction": green_token_count / num_tokens_scored,
            "z_score": z_score,
            "p_value": p_value,
            "prediction": z_score > self.z_threshold
        }