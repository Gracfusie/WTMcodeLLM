#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

_WLLM_DIR = Path(__file__).parent / "third_party" / "WLLM"
if str(_WLLM_DIR) not in sys.path:
    sys.path.insert(0, str(_WLLM_DIR))
try:
    from extended_watermark_processor import WatermarkDetector as WLLMWatermarkDetector
except ImportError:
    print("[Warning] 未找到 WLLM 模块 (extended_watermark_processor)")

_ACW_DIR = Path(__file__).parent / "logits_processors" / "utils"
if str(_ACW_DIR) not in sys.path:
    sys.path.insert(0, str(_ACW_DIR))
try:
    from watermark import ProxyLogitsGuidedWatermarker, ProxyWatermarkDetector
except ImportError:
    print(f"[Warning] 在 {_ACW_DIR} 下未找到 watermark.py 模块，Proxy 模式不可用。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM Chat CLI")
    parser.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=None, help="主模型名称")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=8*1024)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    
    parser.add_argument("--enable-watermark-detection", action="store_true", default=True, help="启用水印检测")
    parser.add_argument("--z-threshold", type=float, default=, help="Z-score 阈值")
    parser.add_argument(
        "--detection-method", 
        choices=["wllm", "proxy"], 
        default=os.environ.get("WATERMARK_ALGORITHM"),
        help="选择检测算法: 'wllm'或 'proxy'"
    )
    parser.add_argument("--proxy-model", default=os.environ.get("WATERMARK_PROXY_MODEL"), help="[Proxy模式] Proxy 模型路径")
    parser.add_argument("--entropy-threshold", type=float, default=os.environ.get("WATERMARK_ENTROPY_THRESHOLD"), help="[Proxy模式] 熵阈值")
    parser.add_argument("--secret-key", type=int, default=os.environ.get("WATERMARK_SECRET_KEY"), help="[Proxy模式] 密钥")

    return parser


def chat_once(client, model, messages, max_tokens, reasoning_effort):
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, reasoning_effort=reasoning_effort
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    watermark_detector = None
    detector_type = "None"

    if args.enable_watermark_detection:
        if args.detection_method == "proxy":
            print(f"[Init] 初始化 Proxy-Guided 检测器...")
            print(f" Proxy Model: {args.proxy_model} | Threshold: {args.entropy_threshold}")
            try:
                try:
                    from modelscope import snapshot_download
                    model_path = snapshot_download(args.proxy_model)
                except ImportError:
                    model_path = args.proxy_model

                proxy_tokenizer = AutoTokenizer.from_pretrained(model_path)
                proxy_model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
                
                core_logic = ProxyLogitsGuidedWatermarker(
                    entropy_threshold=args.entropy_threshold,
                    vocab_size=len(proxy_tokenizer.get_vocab()),
                    secret_key=args.secret_key
                )
                
                # 获取模板配置
                prefix_str = os.environ.get("WATERMARK_PROXY_TEMPLATE_PREFIX", "<|fim_prefix|>")
                suffix_str = os.environ.get("WATERMARK_PROXY_TEMPLATE_SUFFIX", "<|fim_suffix|>\n<|fim_middle|>")
                window_size = int(os.environ.get("WATERMARK_PROXY_WINDOW_SIZE", -1))

                prefix_ids = proxy_tokenizer.encode(prefix_str, add_special_tokens=False)
                suffix_ids = proxy_tokenizer.encode(suffix_str, add_special_tokens=False)
                
                watermark_detector = ProxyWatermarkDetector(
                    watermarker=core_logic,
                    proxy_model=proxy_model,
                    tokenizer=proxy_tokenizer,
                    device=device,
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                    window_size=window_size
                )
                detector_type = "Proxy"
                print("[Init] Proxy 检测器就绪。")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Error] Proxy 检测器初始化失败: {e}")

        elif args.detection_method == "wllm":
            print(f"[Init] 初始化 WLLM (Baseline) 检测器...")
            try:
                model_name = args.model or "Qwen/Qwen2.5-Coder-1.5B-Instruct"
                print(f"       Loading Main Tokenizer: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                watermark_detector = WLLMWatermarkDetector(
                    vocab=list(tokenizer.get_vocab().values()),
                    gamma=0.25
                    delta=2.0,
                    seeding_scheme="selfhash",
                    device=device,
                    tokenizer=tokenizer,
                    z_threshold=args.z_threshold,
                    normalizers=[],
                    ignore_repeated_ngrams=True,
                )
                detector_type = "WLLM"
                print("[Init] WLLM 检测器就绪。")
            except Exception as e:
                print(f"[Error] WLLM 检测器初始化失败: {e}")

    messages = []
    system_prompt = os.environ.get("SYSTEM_PROMPT")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print(f"\n已连接到 {args.endpoint}，主模型 {args.model}")
    print(f"当前检测模式: {detector_type if args.enable_watermark_detection else 'Disabled'}")
    print("输入内容后回车，空行退出。")

    while True:
        try:
            prompt = input("\n你：").strip()
        except EOFError: break
        if not prompt: break
        
        try:
            messages.append({"role": "user", "content": prompt})
            answer = chat_once(client, args.model, messages, args.max_tokens, args.reasoning_effort)
            messages.append({"role": "assistant", "content": answer})
            print(f"模型：{answer}\n")

            if watermark_detector is not None:
                text_to_detect = answer
                if "[Answer]" in answer:
                    text_to_detect = answer.split("[Answer]\n")[-1]

                print(f">>>正在使用 [{detector_type}] 模式检测...")
                try:
                    score_dict = {}
                    if detector_type == "WLLM":
                        score_dict = watermark_detector.detect(
                            text=text_to_detect,
                            return_prediction=True,
                            return_scores=True,
                            z_threshold=args.z_threshold
                        )
                    elif detector_type == "Proxy":
                        score_dict = watermark_detector.detect(
                            text=text_to_detect,
                            z_threshold=args.z_threshold
                        )

                    print(f"[Watermark Report]")
                    
                    n_green = score_dict.get('num_green_tokens', 'N/A')
                    n_total = score_dict.get('num_tokens_scored', 'N/A')
                    z_score = score_dict.get('z_score', 0.0)
                    p_value = score_dict.get('p_value', 'N/A')
                    green_frac = score_dict.get('green_fraction', 'N/A')
                    prediction = score_dict.get('prediction', False)

                    print(f"  - 绿/总 Token: {n_green} / {n_total}")
                    if isinstance(green_frac, float):
                        print(f"  - 绿名单比例 : {green_frac:.3f}")
                    print(f"  - Z-Score    : {z_score:.3f}")
                    if p_value != 'N/A':
                        print(f"  - P-value    : {p_value}")
                    
                    result_icon = '✅' if prediction else '❌'
                    print(f"  - 检测结论   : {result_icon} {'有水印' if prediction else '无水印'}")

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[检测出错] {e}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[系统错误] {e}")
            break
    print("结束。")

if __name__ == "__main__":
    main()