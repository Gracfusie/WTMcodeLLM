#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import json
import time
import argparse
import tokenize
import threading
import random
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
ATTACKS = {"llm_paraphrase", "code_format", "code_truncate", "code_format_decomment"}

# Tokenizer (open-source)
TOKENIZER_NAME = "cl100k_base"
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
# Thread-local client (reduce shared-connection weirdness)
# =========================
_thread_local = threading.local()

def get_thread_client(api_key: str, base_url: str) -> OpenAI:
    c = getattr(_thread_local, "client", None)
    if c is None:
        _thread_local.client = OpenAI(api_key=api_key, base_url=base_url)
        c = _thread_local.client
    return c

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
# Decomment (Python tokenize)
# =========================
def remove_python_comments(code: str) -> str:
    """
    Remove Python line comments using tokenize, preserving whitespace/indentation.
    Keeps strings intact (won't strip '#...' inside strings).
    Returns original code on failure.
    """
    if not isinstance(code, str) or not code.strip():
        return code if isinstance(code, str) else ""

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
        kept = [t for t in toks if t.type != tokenize.COMMENT]
        out = tokenize.untokenize(kept)
        return out
    except Exception:
        return code

def is_python_lang(_: str = "") -> bool:
    # Your code_list items have no language metadata; treat as Python by default.
    return True

# =========================
# Attacks
# =========================
def attack_code_format_one(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return code if isinstance(code, str) else ""
    try:
        return black.format_str(code, mode=black.Mode())
    except black.InvalidInput:
        return code

def attack_code_format_decomment_one(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return code if isinstance(code, str) else ""
    decommented = remove_python_comments(code)
    try:
        return black.format_str(decommented, mode=black.Mode())
    except black.InvalidInput:
        return decommented

def attack_code_truncate_one(code: str) -> str:
    """
    Truncate by proportion: keep the LAST HALF of tokens (token-level, tiktoken).
    """
    if not isinstance(code, str) or not code:
        return code if isinstance(code, str) else ""

    toks = _TOKENIZER.encode(code)
    n = len(toks)
    if n <= 1:
        return code if code.endswith("\n") else code + "\n"

    start = n // 2
    out = _TOKENIZER.decode(toks[start:])
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
            return out if out else code
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** i, 20) + random.random())
    # keep original if totally failed
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
    files: List[str] = []
    root_abs = os.path.abspath(root)
    exclude_abs = os.path.abspath(exclude_dir)

    if recursive:
        for r, dirs, fs in os.walk(root_abs):
            r_abs = os.path.abspath(r)

            if r_abs.startswith(exclude_abs):
                dirs[:] = []
                continue

            dirs[:] = [
                d for d in dirs
                if not os.path.abspath(os.path.join(r_abs, d)).startswith(exclude_abs)
            ]

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
# Progress-bar position allocator (avoid clashes)
# =========================
_pos_lock = threading.Lock()
_next_pos = 1  # 0 reserved for global "Files" pbar
_MAX_ACTIVE_FILE_PBARS = 6  # avoid flooding terminal

def acquire_pbar_position() -> Optional[int]:
    """
    Allocate a tqdm 'position' for per-file pbar.
    If too many active file pbars, return None (we'll disable per-file pbar for that file).
    """
    global _next_pos
    with _pos_lock:
        if _next_pos > _MAX_ACTIVE_FILE_PBARS:
            return None
        pos = _next_pos
        _next_pos += 1
        return pos

def release_pbar_position(pos: Optional[int]) -> None:
    """
    Release a position slot.
    We keep it simple: decrement counter only if releasing last allocated.
    (This is fine for display stability; doesn't affect correctness.)
    """
    global _next_pos
    if pos is None:
        return
    with _pos_lock:
        # best-effort: shrink if releasing highest pos
        if pos == _next_pos - 1 and _next_pos > 1:
            _next_pos -= 1

# =========================
# Core processing
# =========================
def process_one_record_code_list(
    rec: Dict[str, Any],
    attack: str,
    api_key: str,
    base_url: str,
    client_shared: Optional[OpenAI],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    item_workers: int,
    pbar_items: Optional[tqdm],
) -> Dict[str, Any]:
    code_list = rec.get("code_list", None)
    if not isinstance(code_list, list) or not code_list:
        return rec

    codes: List[str] = []
    for x in code_list:
        if isinstance(x, str):
            codes.append(x)
        else:
            codes.append("" if x is None else str(x))

    out_codes: List[Optional[str]] = [None] * len(codes)

    def run_one(i: int) -> Tuple[int, str]:
        c = codes[i]

        if attack == "code_format":
            return i, attack_code_format_one(c)

        if attack == "code_format_decomment":
            if is_python_lang(""):
                return i, attack_code_format_decomment_one(c)
            return i, c

        if attack == "code_truncate":
            return i, attack_code_truncate_one(c)

        if attack == "llm_paraphrase":
            # thread-local client is safer under high concurrency
            tl_client = get_thread_client(api_key, base_url)
            return i, attack_llm_paraphrase_one(
                client=tl_client,
                model=model,
                code=c,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
            )

        raise ValueError(f"Unknown attack: {attack}")

    if item_workers <= 1 or len(codes) == 1:
        for i in range(len(codes)):
            idx, v = run_one(i)
            out_codes[idx] = v
            if pbar_items is not None:
                pbar_items.update(1)
    else:
        with ThreadPoolExecutor(max_workers=item_workers) as ex:
            futures = [ex.submit(run_one, i) for i in range(len(codes))]
            for fut in as_completed(futures):
                idx, v = fut.result()
                out_codes[idx] = v
                if pbar_items is not None:
                    pbar_items.update(1)

    final_codes = [out_codes[i] if out_codes[i] is not None else codes[i] for i in range(len(codes))]
    rec["attack"] = attack
    rec["code_list"] = final_codes
    return rec

def count_items_in_obj(obj: Any) -> int:
    """
    Count total number of code items in this JSON object (sum of lengths of code_list).
    """
    total = 0
    if isinstance(obj, dict):
        cl = obj.get("code_list")
        if isinstance(cl, list):
            total += len(cl)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                cl = it.get("code_list")
                if isinstance(cl, list):
                    total += len(cl)
    return total

def process_json_obj(
    obj: Any,
    attack: str,
    api_key: str,
    base_url: str,
    client_shared: Optional[OpenAI],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    item_workers: int,
    pbar_items: Optional[tqdm],
) -> Any:
    if isinstance(obj, dict):
        return process_one_record_code_list(
            obj, attack, api_key, base_url, client_shared, model,
            temperature, max_tokens, timeout, retries, item_workers, pbar_items
        )

    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict):
                out.append(
                    process_one_record_code_list(
                        item, attack, api_key, base_url, client_shared, model,
                        temperature, max_tokens, timeout, retries, item_workers, pbar_items
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
    api_key: str,
    base_url: str,
    client_shared: Optional[OpenAI],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    item_workers: int,
) -> None:
    obj = load_json(in_path)

    total_items = count_items_in_obj(obj)

    pos = acquire_pbar_position()
    if total_items > 0 and pos is not None:
        pbar_items = tqdm(
            total=total_items,
            desc=f"{os.path.basename(in_path)} items",
            position=pos,
            leave=False,
            dynamic_ncols=True,
        )
    else:
        pbar_items = None

    try:
        out_obj = process_json_obj(
            obj=obj,
            attack=attack,
            api_key=api_key,
            base_url=base_url,
            client_shared=client_shared,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            item_workers=item_workers,
            pbar_items=pbar_items,
        )
    finally:
        if pbar_items is not None:
            pbar_items.close()
        release_pbar_position(pos)

    out_path = map_out_path(in_path, in_dir, out_dir, attack)
    save_json(out_path, out_obj)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--attack", required=True, choices=sorted(list(ATTACKS)))
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", OPENROUTER_BASE_URL))
    ap.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview"))

    ap.add_argument("--file-workers", type=int, default=4)
    ap.add_argument("--item-workers", type=int, default=8)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=4)

    args = ap.parse_args()

    if args.attack == "llm_paraphrase" and not args.api_key:
        raise SystemExit("Missing API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")

    client_shared = OpenAI(api_key=args.api_key, base_url=args.base_url) if args.attack == "llm_paraphrase" else None

    files = iter_json_files(args.in_dir, args.recursive, exclude_dir=args.out_dir)
    if not files:
        raise SystemExit(f"No .json files found in {args.in_dir} (excluding {args.out_dir})")

    # Outer/global pbar: files
    with tqdm(total=len(files), desc=f"Files ({args.attack})", position=0, dynamic_ncols=True) as pbar_files:
        with ThreadPoolExecutor(max_workers=args.file_workers) as ex:
            futures = [
                ex.submit(
                    process_one_file,
                    in_path,
                    args.in_dir,
                    args.out_dir,
                    args.attack,
                    args.api_key,
                    args.base_url,
                    client_shared,
                    args.model,
                    args.temperature,
                    args.max_tokens,
                    args.timeout,
                    args.retries,
                    args.item_workers,
                )
                for in_path in files
            ]
            for fut in as_completed(futures):
                fut.result()
                pbar_files.update(1)

    print(
        f"Done. attack={args.attack} files={len(files)} "
        f"file_workers={args.file_workers} item_workers={args.item_workers} "
        f"-> {args.out_dir}"
    )

if __name__ == "__main__":
    main()
