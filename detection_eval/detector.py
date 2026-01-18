#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from hashlib import sha256

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_WLLM_DIR = Path(__file__).parent / "third_party" / "WLLM"
if str(_WLLM_DIR) not in sys.path:
    sys.path.insert(0, str(_WLLM_DIR))

from logits_processors.utils.sweet_detector import SweetDetector
from logits_processors.utils.watermark import ProxyLogitsGuidedWatermarker, ProxyWatermarkDetector
from config import EnvConfig

class Detector:
    def __init__(
        self,
        z_threshold: float,
        method: str = "proxy",
        device: Optional[torch.device] = None,
        logits_model: Optional[str] = None, # 自动推断
        entropy_threshold: float = EnvConfig.watermark_entropy_threshold,
        secret_key: int = EnvConfig.watermark_secret_key,
        gamma: float = EnvConfig.watermark_gamma,
        delta: float = EnvConfig.watermark_delta,
        prefix_str: str = EnvConfig.watermark_proxy_template_prefix,
        suffix_str: str = EnvConfig.watermark_proxy_template_suffix,
        window_size: int = EnvConfig.window_size,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if logits_model is None:
            if method == "proxy":
                logits_model = EnvConfig.watermark_proxy_model
            elif method == "wllm":
                logits_model = EnvConfig.main_model
            else:
                raise NotImplementedError(f"Unknown method: {method}")
          

        self.method = method
        self.logits_model = logits_model
        self.entropy_threshold = entropy_threshold
        self.secret_key = secret_key
        self.gamma = gamma
        self.delta = delta
        self.prefix_str = prefix_str
        self.suffix_str = suffix_str
        self.window_size = window_size

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
        self.tokenizer = None
        self.z_threshold = z_threshold

        if method == "wllm":
            self.tokenizer = AutoTokenizer.from_pretrained(EnvConfig.main_model)
            self.detector = SweetDetector(
                entropy_threshold=-1., # 全部加水印
                hash_key=secret_key,
                vocab_size=len(self.tokenizer.get_vocab()),
                gamma=gamma,
                z_threshold=z_threshold,
                tokenizer=self.tokenizer,
            )

        elif method == "proxy":
            if os.environ.get('VLLM_USE_MODELSCOPE', '1') != '0':
                from modelscope import snapshot_download
                model_path = snapshot_download(logits_model)
            else:
                model_path = logits_model

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            proxy_model_obj = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto"
            ).to(self.device).eval()

            core_logic = ProxyLogitsGuidedWatermarker(
                entropy_threshold=entropy_threshold,
                vocab_size=len(self.tokenizer.get_vocab()),
                secret_key=secret_key,
            )

            prefix_ids = self.tokenizer.encode(prefix_str, add_special_tokens=False)
            suffix_ids = self.tokenizer.encode(suffix_str, add_special_tokens=False)

            self.detector = ProxyWatermarkDetector(
                watermarker=core_logic,
                proxy_model=proxy_model_obj,
                tokenizer=self.tokenizer,
                device=self.device,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                window_size=window_size,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def detect(self, texts: List[str]) -> List[dict]:
        if self.method == "wllm":
            score_dicts = self.detector.detect_without_model(
                text=texts,
            )
        elif self.method == "proxy":
            score_dicts = self.detector.detect(
                text=texts,
                z_threshold=self.z_threshold,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return score_dicts

def sanity_check():
    texts = [
        "This is a test sentence.",
        "Another example text to detect.",
        # 我们方法生成的字符串
        """I'm an AI assistant created by Alibaba Cloud, and my name is Qwen. I can answer questions, create text, play games, and provide help in many areas. My goal is to assist users in achieving their desired outcomes and provide comprehensive information and优质的服务. As an AI model, I have a vast amount of knowledge and can answer questions on various topics, including but not limited to science, technology, culture, and life.

My favorite movie is "Inception" (《盗梦空间》). This film, directed by Christopher Nolan, is a masterpiece that explores the concept of dreams and reality. It combines stunning visuals with a complex storyline, making it both thrilling and thought-provoking. The movie delves into the depths of human consciousness and the power of imagination. It's an excellent choice for those who enjoy movies with deep meanings and visual effects.

I hope you will like this introduction to me! If you have any other questions, feel free to ask anytime!""",
        # wllm 生成的
        """# Read input
X = int(input())

# Calculate factorial iteratively until we find N such that N! = X
factorial = 1
N = 1

# Keep increasing N until factorial equals X
while factorial < X:
    N += 1
    factorial *= N

# At this point, factorial equals X
print(N)"""
    ]
    
    results = detector.detect(texts)
    for i, r in enumerate(results):
        print(f"Text {i}: z_score={r['z_score']:.3f}, info={r}")


def str_hash(x: str) -> str:
    return sha256(x.encode()).hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="proxy")
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--input-dir", type=str, default="input/")
    parser.add_argument("--output-dir", type=str, default="output/")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    detector = Detector(
        method=args.method,
        z_threshold=args.z_threshold,
    )
    sanity_check()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    pending = []
    for file in Path(args.input_dir).glob("*.json"):
        data = json.load(open(file, "r"))
        for item in data:
            for code in item['code_list']:
                if Path(args.output_dir, f"{str_hash(code)}.json").exists():
                    continue
                pending.append(code)

    output_dict = {}
    for i in tqdm(range(0, len(pending), args.batch_size)):
        batch = pending[i:i+args.batch_size]
        results = detector.detect(batch)
        for j, result in enumerate[dict](results):
            Path(args.output_dir, f"{str_hash(batch[j])}.json").write_text(json.dumps(result, indent=4))
            output_dict[batch[j]] = result
    
    # with open(Path(args.output_dir, "zscores.json"), "w") as f:
        # json.dump(output_dict, f, indent=4)
