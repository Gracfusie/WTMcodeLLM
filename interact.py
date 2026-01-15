#!/usr/bin/env python3

import argparse
import os
from openai import OpenAI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM Chat CLI")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"),
        help="OpenAI 兼容 API 地址，默认 http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称或路径，默认 None",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key（本地可用占位值）",
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
        "--watermark-entropy-threshold",
        type=float,
        default=None,
        help="override watermark_entropy_threshold",
    )
    parser.add_argument(
        "--watermark-delta",
        type=float,
        default=None,
        help="override watermark_delta",
    )
    parser.add_argument(
        "--watermark-window-size",
        type=int,
        default=None,
        help="override watermark_proxy_window_size",
    )
    return parser


def chat_once(
    client: OpenAI,
    model: str,
    messages: list,
    max_tokens: int,
    reasoning_effort: str,
    extra_body: dict = None,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    content = msg.content
    if reasoning:
        return f"[Reasoning]\n{reasoning}\n\n[Answer]\n{content}"
    return content


def main():
    args = build_parser().parse_args()
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    extra_args = {}
    if args.watermark_entropy_threshold is not None:
        extra_args["watermark_entropy_threshold"] = args.watermark_entropy_threshold
    if args.watermark_delta is not None:
        extra_args["watermark_delta"] = args.watermark_delta
    if args.watermark_window_size is not None:
        extra_args["watermark_proxy_window_size"] = args.watermark_window_size
    
    if extra_args:
        print(f"使用 override 参数: {extra_args}")

    messages = []
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
                args.max_tokens,
                args.reasoning_effort,
                {"vllm_xargs": extra_args} if extra_args else None,
            )
            messages.append({"role": "assistant", "content": answer})
            print(f"模型：{answer}\n")
        except Exception as e:
            print(f"[错误] {e}")
            break
    print("结束。")


if __name__ == "__main__":
    main()

