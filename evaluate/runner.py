#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any

from openai import OpenAI

# Ensure we can import chat_once from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from interact import chat_once  # type: ignore
from evaluate.attacks import apply_attack, AttackName
from evaluate.base import DummyZScoreDetector, ProxyDetector
from evaluate.watermark_params import WatermarkParams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watermark Detector Runner")

    # Model and API settings (aligned with interact.py)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API endpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key (placeholder for local)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8 * 1024,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=os.environ.get("REASONING_EFFORT", "high"),
        help="Reasoning effort",
    )

    # Watermark parameter overrides (forwarded via extra_args)
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

    # Proxy detector specific controls
    parser.add_argument(
        "--proxy-model",
        type=str,
        default=os.environ.get("WATERMARK_PROXY_MODEL"),
        help="Proxy model path or hub id for proxy-guided detector",
    )
    parser.add_argument(
        "--proxy-secret-key",
        type=int,
        default=os.environ.get("WATERMARK_SECRET_KEY"),
        help="Secret key for proxy watermarker",
    )
    parser.add_argument(
        "--proxy-template-prefix",
        type=str,
        default=os.environ.get("WATERMARK_PROXY_TEMPLATE_PREFIX", "<|fim_prefix|>"),
        help="Prefix template for proxy detector",
    )
    parser.add_argument(
        "--proxy-template-suffix",
        type=str,
        default=os.environ.get("WATERMARK_PROXY_TEMPLATE_SUFFIX", "<|fim_suffix|>\n<|fim_middle|>"),
        help="Suffix template for proxy detector",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=4.0,
        help="Z-score threshold used by proxy detector",
    )

    # Parameter grid controls
    parser.add_argument(
        "--entropy-list",
        type=str,
        default=None,
        help="Comma-separated list of entropy thresholds (e.g., 2.0,3.0)",
    )
    parser.add_argument(
        "--delta-list",
        type=str,
        default=None,
        help="Comma-separated list of delta values (e.g., 0.1,0.2)",
    )
    parser.add_argument(
        "--window-list",
        type=str,
        default=None,
        help="Comma-separated list of window sizes (e.g., 128,256)",
    )
    parser.add_argument(
        "--grid-default",
        action="store_true",
        help="Run a small default parameter sweep when no lists provided",
    )

    # Detector and I/O controls
    parser.add_argument(
        "--attack",
        choices=["none", "truncate", "reconstruct", "normalize"],
        default="none",
        help="Simulated attack to apply to the generated sample",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.environ.get("DETECTOR_INPUT_DIR", "input"),
        help="Directory containing prompt files (default: input)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to evaluate (overrides input-dir)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to a file containing prompts (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("DETECTOR_OUTPUT_DIR", "output"),
        help="Output directory for results (default: output); overridden by extra_args if present",
    )
    return parser


def load_prompts(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Return list of {text, stem} for naming outputs."""
    items: List[Dict[str, Any]] = []
    # Highest precedence: explicit single prompt
    if args.prompt:
        items.append({"text": args.prompt, "stem": "prompt_cli"})
        return items
    # Next: prompts file, one per line
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                ln = line.strip()
                if ln:
                    items.append({"text": ln, "stem": f"prompts_file_{i:03d}"})
        if items:
            return items
    # Default: scan input directory and read whole file per prompt
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input dir not found: {input_dir}")
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        if not any(name.endswith(ext) for ext in (".txt", ".md")):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                stem, _ = os.path.splitext(name)
                items.append({"text": text, "stem": stem})
    if not items:
        raise ValueError("No prompts found in input dir; add .txt/.md files or use --prompt/--prompts-file.")
    return items


def build_extra_args(args: argparse.Namespace) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if args.watermark_entropy_threshold is not None:
        extra["watermark_entropy_threshold"] = args.watermark_entropy_threshold
    if args.watermark_delta is not None:
        extra["watermark_delta"] = args.watermark_delta
    if args.watermark_window_size is not None:
        extra["watermark_proxy_window_size"] = args.watermark_window_size
    # Set output_dir in extra_args so upstream can inspect/use it
    extra["output_dir"] = args.output_dir
    return extra


def _parse_float_list(s: Optional[str]) -> Optional[List[Optional[float]]]:
    if not s:
        return None
    vals: List[Optional[float]] = []
    for part in s.split(","):
        p = part.strip()
        if p.lower() in ("na", "none", ""):
            vals.append(None)
        else:
            vals.append(float(p))
    return vals


def _parse_int_list(s: Optional[str]) -> Optional[List[Optional[int]]]:
    if not s:
        return None
    vals: List[Optional[int]] = []
    for part in s.split(","):
        p = part.strip()
        if p.lower() in ("na", "none", ""):
            vals.append(None)
        else:
            vals.append(int(p))
    return vals


def build_param_grid(args: argparse.Namespace) -> List[WatermarkParams]:
    ent_list = _parse_float_list(args.entropy_list)
    d_list = _parse_float_list(args.delta_list)
    w_list = _parse_int_list(args.window_list)

    # Default small sweep if requested and no lists provided
    if not ent_list and not d_list and not w_list and args.grid_default:
        ent_list = [2.0, 3.0]
        d_list = [0.1, 0.2]
        w_list = [128, 256]

    # If no lists provided and no default grid, return single param set possibly from singular overrides
    if not ent_list and not d_list and not w_list:
        return [WatermarkParams(
            entropy_threshold=args.watermark_entropy_threshold,
            delta=args.watermark_delta,
            proxy_window_size=args.watermark_window_size,
        )]

    # Use [None] for missing lists to include NA in combinations
    ent_list = ent_list or [None]
    d_list = d_list or [None]
    w_list = w_list or [None]

    grid: List[WatermarkParams] = []
    for ent in ent_list:
        for dlt in d_list:
            for win in w_list:
                grid.append(WatermarkParams(
                    entropy_threshold=ent,
                    delta=dlt,
                    proxy_window_size=win,
                ))
    return grid


def resolve_output_dir(cli_output_dir: str, extra_args: Optional[Dict[str, Any]]) -> str:
    # Prefer extra_args-provided directory when present
    if extra_args and isinstance(extra_args, dict):
        out = extra_args.get("output_dir")
        if isinstance(out, str) and out.strip():
            return out.strip()
    return cli_output_dir


def save_result(output_dir: str, result: Dict[str, Any], stem: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"{stem}.{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def build_output_stem(
    base_stem: str,
    model: Optional[str],
    attack: str,
    params: WatermarkParams,
    algorithm: Optional[str] = None,
) -> str:
    def sanitize(s: Optional[str]) -> str:
        if not s:
            return "nomodel"
        return (
            s.replace("/", "-")
            .replace(":", "-")
            .replace(" ", "-")
        )

    m = sanitize(model)
    ent = params.entropy_threshold
    dlt = params.delta
    win = params.proxy_window_size
    ent_s = f"ent{ent}" if ent is not None else "entNA"
    dlt_s = f"d{dlt}" if dlt is not None else "dNA"
    win_s = f"w{win}" if win is not None else "wNA"
    alg = algorithm or os.environ.get("WATERMARK_ALGORITHM", "unknown")
    alg = alg.replace("/", "-").replace(":", "-").replace(" ", "-")
    return f"{base_stem}.{m}.{attack}.{ent_s}.{dlt_s}.{win_s}.alg{alg}"


def run() -> None:
    args = build_parser().parse_args()
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    extra_args = build_extra_args(args)
    output_dir = resolve_output_dir(args.output_dir, extra_args)

    if extra_args:
        print(f"使用 override 参数: {extra_args}")
    print(f"结果输出到: {output_dir}")

    system_prompt = os.environ.get("SYSTEM_PROMPT")
    prompts = load_prompts(args)
    try:
        detector = ProxyDetector(
            proxy_model=args.proxy_model,
            entropy_threshold_default=args.watermark_entropy_threshold,
            secret_key=args.proxy_secret_key,
            window_size=args.watermark_window_size,
            z_threshold=args.z_threshold,
            prefix_str=args.proxy_template_prefix,
            suffix_str=args.proxy_template_suffix,
        )
        print("[Init] Proxy detector ready.")
    except Exception as e:
        print(f"[警告] Proxy detector init failed, fallback to DummyZScoreDetector: {e}")
        detector = DummyZScoreDetector()
    param_grid = build_param_grid(args)

    for idx, item in enumerate(prompts):
        # Prepare messages with optional system prompt (as in interact.py)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": item["text"]})

        for pidx, wm_params in enumerate(param_grid):
            # Merge per-run params into extra_args
            xargs = dict(extra_args)
            if wm_params.entropy_threshold is not None:
                xargs["watermark_entropy_threshold"] = wm_params.entropy_threshold
            else:
                xargs.pop("watermark_entropy_threshold", None)
            if wm_params.delta is not None:
                xargs["watermark_delta"] = wm_params.delta
            else:
                xargs.pop("watermark_delta", None)
            if wm_params.proxy_window_size is not None:
                xargs["watermark_proxy_window_size"] = wm_params.proxy_window_size
            else:
                xargs.pop("watermark_proxy_window_size", None)

            try:
                answer = chat_once(
                    client=client,
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    extra_body={"vllm_xargs": xargs} if xargs else None,
                )
            except Exception as e:
                print(f"[错误] 生成样本失败 (pidx={pidx}): {e}")
                continue

            attacked = apply_attack(answer, args.attack)  # type: ignore[arg-type]
            z = detector.detect(attacked, wm_params)

            result = {
                "index": idx,
                "param_index": pidx,
                "prompt": item["text"],
                "attack": args.attack,
                "sample_raw": answer,
                "sample_attacked": attacked,
                "z_score": z,
                "watermark_algorithm": (extra_args.get("watermark_algorithm") if extra_args and "watermark_algorithm" in extra_args else os.environ.get("WATERMARK_ALGORITHM", "unknown")),
                "watermark_params": {
                    "entropy_threshold": wm_params.entropy_threshold,
                    "delta": wm_params.delta,
                    "proxy_window_size": wm_params.proxy_window_size,
                },
            }

            algorithm = result["watermark_algorithm"]
            stem = build_output_stem(item["stem"], args.model, args.attack, wm_params, algorithm=algorithm)
            out_path = save_result(output_dir, result, stem)
            print(f"完成: z={z:.4f} -> {out_path}")


if __name__ == "__main__":
    run()
