#!/usr/bin/env python3
"""
简单 CLI：与本地 vLLM OpenAI 兼容端点交互，验证 gpt-oss-20b 部署。
默认采样参数 temperature=1.0, top_p=1.0。
"""

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
        default=os.environ.get("MODEL_NAME", "openai/gpt-oss-20b"),
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
        default=32768,
        help="回复最大生成 token 数，默认 32768",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=os.environ.get("REASONING_EFFORT", "high"),
        help="reasoning_effort，默认 high，可选 low/medium/high",
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
        except Exception as e:
            print(f"[错误] {e}")
            break
    print("结束。")


if __name__ == "__main__":
    main()

