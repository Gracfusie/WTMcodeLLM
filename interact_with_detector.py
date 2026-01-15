#!/usr/bin/env python3
"""
简单 CLI：与本地 vLLM OpenAI 兼容端点交互，验证 gpt-oss-20b 部署。
默认采样参数 temperature=1.0, top_p=1.0。
"""

import argparse
import os

from openai import OpenAI

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from config import EnvConfig

# 加载 WLLM watermark detector
_WLLM_DIR = Path(__file__).parent / "third_party" / "WLLM"
if str(_WLLM_DIR) not in sys.path:
    sys.path.insert(0, str(_WLLM_DIR))

from extended_watermark_processor import WatermarkDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM Chat CLI")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"),
        help="OpenAI 兼容 API 地址，默认 http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MAIN_MODEL", "openai/gpt-oss-20b"),
        help="模型名称或路径，默认 openai/gpt-oss-20b",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key（本地可用占位值）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样 temperature，默认 1.0",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="采样 top_p，默认 1.0",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8*1024,
        help="回复最大生成 token 数，默认 8k",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=os.environ.get("REASONING_EFFORT", "high"),
        help="reasoning_effort，默认 high，可选 low/medium/high",
    )
    parser.add_argument(
        "--enable-watermark-detection",
        action="store_true",
        default=False,
        help="启用水印检测",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=4.0,
        help="水印检测 z-score 阈值，默认 4.0",
    )
    return parser


def chat_once(
    client: OpenAI,
    model: str,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    msg = resp.choices[0].message
    # 兼容包含 reasoning_content 的模型，同时保留最终回答
    reasoning = getattr(msg, "reasoning_content", None)
    content = msg.content
    if reasoning:
        return f"[Reasoning]\n{reasoning}\n\n[Answer]\n{content}"
    return content


def main():
    args = build_parser().parse_args()
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    # 初始化水印检测器（如果启用）
    watermark_detector = None
    if args.enable_watermark_detection:
        try:
            # 加载 tokenizer（对应 gpt-oss-20b）
            print("[Watermark] 加载 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
            print("[Watermark] tokenizer 加载完成")

            # 初始化检测器，参数与 serve.sh 中的 WatermarkVLLMAdapter 保持一致
            # 设备与 vLLM 服务端保持一致（优先 GPU，保证 RNG 一致性）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[Watermark] 使用设备: {device}")
            watermark_detector = WatermarkDetector(
                vocab=list(tokenizer.get_vocab().values()),
                gamma=0.25,  # 与 watermark_adapter.py 保持一致
                delta=2.0,
                seeding_scheme="selfhash",  # 与 watermark_adapter.py 保持一致
                device=device,
                tokenizer=tokenizer,
                z_threshold=args.z_threshold,
                normalizers=[],
                ignore_repeated_bigrams=True,
            )
            print("[Watermark] 检测器初始化完成\n")
        except Exception as e:
            print(f"[Watermark] 检测器初始化失败: {e}")
            watermark_detector = None

    messages = []
    # 可以通过环境变量提供初始 system 提示，便于对话控制
    system_prompt = os.environ.get("SYSTEM_PROMPT")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print(f"已连接到 {args.endpoint}，模型 {args.model}")
    print("输入内容后回车，空行退出（保留多轮上下文）。")
    while True:
        try:
            prompt = input("你：").strip()
        except EOFError:
            break
        if not prompt:
            break
        try:
            messages.append({"role": "user", "content": prompt})
            answer = chat_once(
                client,
                args.model,
                messages,
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.reasoning_effort,
            )
            messages.append({"role": "assistant", "content": answer})
            print(f"模型：{answer}\n")

            # 水印检测（如果启用）
            if watermark_detector is not None:
                try:
                    # 仅对 [Answer] 部分检测（去掉 [Reasoning] 部分如果有）
                    text_to_detect = answer
                    if "[Answer]" in answer:
                        text_to_detect = answer.split("[Answer]\n")[-1]

                    score_dict = watermark_detector.detect(
                        text=text_to_detect,
                        return_prediction=True,
                        return_scores=True,
                    )
                    print(f"[Watermark] 检测结果:")
                    print(f"  - 绿 token 数: {score_dict.get('num_green_tokens', 'N/A')}")
                    print(f"  - 总 token 数: {score_dict.get('num_tokens_scored', 'N/A')}")
                    print(f"  - 绿 token 比例: {score_dict.get('green_fraction', 'N/A'):.3f}")
                    print(f"  - Z-score: {score_dict.get('z_score', 'N/A'):.3f}")
                    print(f"  - P-value: {score_dict.get('p_value', 'N/A'):.6f}")
                    prediction = score_dict.get('prediction', False)
                    confidence = score_dict.get('confidence', 0.0)
                    print(f"  - 检测结果: {'✓ 有水印' if prediction else '✗ 无水印'} (置信度: {confidence:.3f})\n")
                except Exception as e:
                    print(f"[Watermark] 检测失败: {e}\n")
        except Exception as e:
            print(f"[错误] {e}")
            break
    print("结束。")


if __name__ == "__main__":
    main()

