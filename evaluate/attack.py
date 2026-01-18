#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI
import black
import tiktoken

# =========================
# Defaults / constants
# =========================
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ATTACKS = {"llm_paraphrase", "code_format", "code_truncate"}

# Truncate settings (token-level, open-source)
TOKENIZER_NAME = "cl100k_base"
TRUNCATE_TOKENS = 1024
_TOKENIZER = tiktoken.get_encoding(TOKENIZER_NAME)

# Optional: extract fenced code if the model returns it
FENCED_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*([a-zA-Z0-9_+\-\.]*)[ \t]*\n(.*?)\n```",
    re.DOTALL,
)

SYSTEM_PROMPT_PARAPHRASE_CODE = (
    "You are a code rewriting engine.\n"
    "Rewrite the given code while preserving exact functionality.\n"
    "Constraints:\n"
    "- Output ONLY the rewritten code.\n"
    "- Do NOT include explanations, comments about what you did, or markdown fences.\n"
    "- Keep the same language.\n"
)


# =========================
# Helpers: code extraction
# =========================
def strip_to_code_only(text: str) -> str:
    """
    If text contains fenced code blocks, return concatenated code contents (without fences).
    Otherwise return text as-is (trimmed).
    """
    if not text:
        return ""
    matches = list(FENCED_CODE_BLOCK_RE.finditer(text))
    if not matches:
        return text.strip()

    parts = []
    for m in matches:
        code = m.group(2) or ""
        parts.append(code)
    out = "\n\n".join(parts).strip()
    return out


# =========================
# Attacks
# =========================
def attack_code_format_one(code: str) -> str:
    """
    Open-source formatting using Black for Python.
    If not valid Python snippet for black, returns original unchanged.
    """
    if not isinstance(code, str) or not code.strip():
        return code if isinstance(code, str) else ""
    try:
        return black.format_str(code, mode=black.Mode())
    except black.InvalidInput:
        return code


def attack_code_truncate_one(code: str) -> str:
    """
    Token-level truncation using tiktoken(cl100k_base).
    Keeps first TRUNCATE_TOKENS tokens.
    """
    if not isinstance(code, str) or not code:
        return code if isinstance(code, str) else ""
    toks = _TOKENIZER.encode(code)
    if len(toks) <= TRUNCATE_TOKENS:
        return code if code.endswith("\n") else code + "\n"
    out = _TOKENIZER.decode(toks[:TRUNCATE_TOKENS])
    return out if out.endswith("\n") else out + "\n"


def attack_llm_paraphrase_one(
    client: OpenAI,
    model: str,
    code: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    """
    LLM paraphrase for CODE (not prose). Enforces code-only output.
    If model still returns fences, strips them.
    """
    if not isinstance(code, str) or not code.strip():
        return code if isinstance(code, str) else ""

    last_err = None
    for i in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_PARAPHRASE_CODE},
                    {"role": "user", "content": code},
                ],
                timeout=timeout,
            )
            out = (resp.choices[0].message.content or "").strip()
            out = strip_to_code_only(out)
            # fallback if somehow empty
            return out if out else code
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** i, 20))
    # if totally failed, keep original to avoid breaking pipeline
    return code


# =========================
# IO / file traversal
# =========================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def map_out_path(in_path: str, in_dir: str, out_dir: str, attack: str) -> str:
    rel = os.path.relpath(in_path, in_dir)
    base, ext = os.path.splitext(rel)
    return os.path.join(out_dir, f"{base}.{attack}{ext}")


def iter_json_files(root: str, recursive: bool, exclude_dir: str) -> List[str]:
    """
    Iterate .json files under root, excluding anything under exclude_dir.
    This prevents re-attacking already-written outputs when out_dir is inside in_dir.
    """
    files: List[str] = []
    root_abs = os.path.abspath(root)
    exclude_abs = os.path.abspath(exclude_dir)

    if recursive:
        for r, dirs, fs in os.walk(root_abs):
            r_abs = os.path.abspath(r)

            # prune traversal if inside exclude_dir
            if r_abs.startswith(exclude_abs):
                dirs[:] = []
                continue

            # Also prune dirs that would enter exclude_dir
            # (helps when exclude_dir is a subfolder of root)
            dirs[:] = [d for d in dirs if not os.path.abspath(os.path.join(r_abs, d)).startswith(exclude_abs)]

            for fn in fs:
                if fn.endswith(".json"):
                    p = os.path.join(r_abs, fn)
                    if not os.path.abspath(p).startswith(exclude_abs):
                        files.append(p)
    else:
        for fn in os.listdir(root_abs):
            if fn.endswith(".json"):
                p = os.path.join(root_abs, fn)
                if not os.path.abspath(p).startswith(exclude_abs):
                    files.append(p)

    return sorted(files)


# =========================
# Core processing
# =========================
def process_one_record_code_list(
    rec: Dict[str, Any],
    attack: str,
    client: Optional[OpenAI],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    item_workers: int,
) -> Dict[str, Any]:
    """
    Mutates one record:
      - reads rec["code_list"] (list[str])
      - replaces it with attacked code_list
      - sets rec["attack"] = attack
    """
    code_list = rec.get("code_list", None)
    if not isinstance(code_list, list) or not code_list:
        return rec

    # ensure all are strings (coerce non-str to str or keep as-is)
    codes: List[str] = []
    for x in code_list:
        if isinstance(x, str):
            codes.append(x)
        else:
            # keep stable: stringify non-str rather than crashing
            codes.append("" if x is None else str(x))

    out_codes: List[Optional[str]] = [None] * len(codes)

    def run_one(i: int) -> Tuple[int, str]:
        c = codes[i]
        if attack == "code_format":
            return i, attack_code_format_one(c)
        if attack == "code_truncate":
            return i, attack_code_truncate_one(c)
        if attack == "llm_paraphrase":
            assert client is not None
            return i, attack_llm_paraphrase_one(
                client=client,
                model=model,
                code=c,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
            )
        raise ValueError(f"Unknown attack: {attack}")

    # item-level parallelism
    if item_workers <= 1 or len(codes) == 1:
        for i in range(len(codes)):
            idx, v = run_one(i)
            out_codes[idx] = v
    else:
        with ThreadPoolExecutor(max_workers=item_workers) as ex:
            futures = [ex.submit(run_one, i) for i in range(len(codes))]
            for fut in as_completed(futures):
                idx, v = fut.result()
                out_codes[idx] = v

    # fill any missing (shouldn't happen)
    final_codes = [out_codes[i] if out_codes[i] is not None else codes[i] for i in range(len(codes))]

    rec["attack"] = attack
    rec["code_list"] = final_codes
    return rec


def process_json_obj(
    obj: Any,
    attack: str,
    client: Optional[OpenAI],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    item_workers: int,
) -> Any:
    """
    Supports:
      - list[dict] : process each dict (sequentially; code_list inside is parallel)
      - dict       : process dict
      - others     : passthrough
    """
    if isinstance(obj, dict):
        return process_one_record_code_list(
            obj, attack, client, model, temperature, max_tokens, timeout, retries, item_workers
        )

    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict):
                out.append(
                    process_one_record_code_list(
                        item, attack, client, model, temperature, max_tokens, timeout, retries, item_workers
                    )
                )
            else:
                out.append(item)
        return out

    return obj


def process_one_file(
    in_path: str,
    in_dir: str,
    out_dir: str,
    attack: str,
    client: Optional[OpenAI],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    item_workers: int,
) -> None:
    obj = load_json(in_path)
    out_obj = process_json_obj(
        obj=obj,
        attack=attack,
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        item_workers=item_workers,
    )
    out_path = map_out_path(in_path, in_dir, out_dir, attack)
    save_json(out_path, out_obj)


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--attack", required=True, choices=sorted(list(ATTACKS)))
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", OPENROUTER_BASE_URL))
    ap.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder-30b-a3b-instruct"))

    # parallelism
    ap.add_argument("--file-workers", type=int, default=4, help="Number of JSON files processed in parallel")
    ap.add_argument("--item-workers", type=int, default=8, help="Number of code items per record processed in parallel")

    # llm params (used only for llm_paraphrase)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=4)

    args = ap.parse_args()

    if args.attack == "llm_paraphrase" and not args.api_key:
        raise SystemExit("Missing API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")

    # Create one client shared by threads (safe for typical HTTP usage)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url) if args.attack == "llm_paraphrase" else None

    files = iter_json_files(args.in_dir, args.recursive, exclude_dir=args.out_dir)
    if not files:
        raise SystemExit(f"No .json files found in {args.in_dir} (excluding {args.out_dir})")

    # file-level parallelism
    with ThreadPoolExecutor(max_workers=args.file_workers) as ex:
        futures = [
            ex.submit(
                process_one_file,
                in_path,
                args.in_dir,
                args.out_dir,
                args.attack,
                client,
                args.model,
                args.temperature,
                args.max_tokens,
                args.timeout,
                args.retries,
                args.item_workers,
            )
            for in_path in files
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Attack={args.attack}"):
            # Raise exceptions early if any
            fut.result()

    print(
        f"Done. attack={args.attack} files={len(files)} "
        f"file_workers={args.file_workers} item_workers={args.item_workers} "
        f"-> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
