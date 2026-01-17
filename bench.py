#!/usr/bin/env python3

import argparse
import os
import time
import asyncio
from uuid import uuid4
from openai import AsyncOpenAI


async def single_request(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> dict:
    """单次请求，返回耗时和 token 数"""
    start = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start
    output_tokens = resp.usage.completion_tokens
    return {"elapsed": elapsed, "tokens": output_tokens}


async def benchmark(endpoint: str, api_key: str, model: str, prompt: str, 
                    max_tokens: int, concurrency: int, num_requests: int):
    """并发测试"""
    client = AsyncOpenAI(base_url=endpoint, api_key=api_key)
    
    print(f"端点: {endpoint}")
    print(f"模型: {model}")
    print(f"并发: {concurrency}, 总请求: {num_requests}, max_tokens: {max_tokens}")
    print("-" * 50)
    
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    total_tokens = 0
    total_latency = 0
    start = time.perf_counter()
    
    async def limited_request():
        nonlocal completed, total_tokens, total_latency
        async with semaphore:
            result = await single_request(client, model, str(uuid4()) + prompt, max_tokens)
            completed += 1
            total_tokens += result["tokens"]
            total_latency += result["elapsed"]
            
            # 实时输出进度
            elapsed = time.perf_counter() - start
            avg_latency = total_latency / completed
            current_qps = completed / elapsed if elapsed > 0 else 0
            current_throughput = total_tokens / elapsed if elapsed > 0 else 0
            single_throughput = result['tokens'] / result['elapsed'] if result['elapsed'] > 0 else 0
            
            print(f"[{completed}/{num_requests}] "
                  f"延迟: {result['elapsed']:.2f}s | "
                  f"tokens: {result['tokens']} | "
                  f"单请求: {single_throughput:.2f} tok/s | "
                  f"总体: {current_throughput:.2f} tok/s | "
                  f"QPS: {current_qps:.2f} | "
                  f"平均延迟: {avg_latency:.2f}s", flush=True)
            
            return result
    
    results = await asyncio.gather(*[limited_request() for _ in range(num_requests)])
    total_time = time.perf_counter() - start
    
    print("-" * 50)
    print(f"✓ 测试完成！")
    print(f"总耗时: {total_time:.2f}s")
    print(f"总输出 tokens: {total_tokens}")
    print(f"吞吐量: {total_tokens / total_time:.2f} tokens/s")
    print(f"QPS: {num_requests / total_time:.2f}")
    print(f"平均延迟: {total_latency / len(results):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="API 吞吐量测试")
    parser.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--prompt", default="写一个 100 行左右的 py 小游戏～要有创意！")
    parser.add_argument("--max-tokens", type=int, default=1024*16)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=256)
    args = parser.parse_args()
    
    asyncio.run(benchmark(
        args.endpoint, args.api_key, args.model, args.prompt,
        args.max_tokens, args.concurrency, args.num_requests
    ))


if __name__ == "__main__":
    main()