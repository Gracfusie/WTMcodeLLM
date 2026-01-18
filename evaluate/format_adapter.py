#!/usr/bin/env python3

"""
Utility to convert mid-stage JSON results ("midput") into per-sample output
JSON files compatible with analyze_outputs.py.

Usage example:
	python format_adapter.py \
		--input midput/Scenario.codegeneration_1_0.7_eval_all.json \
		--output-dir output/converted \
		--base-stem scenario

Fields produced per output JSON:
	index              : zero-based global index within this conversion run
	prompt             : concatenated question title + content
	attack             : passthrough from CLI (default "none")
	sample_raw         : raw model output from midput
	sample_attacked    : same as raw (no attack applied here)
	z_score            : null placeholder (no detector run at this stage)
	watermark_algorithm: configurable label (default "unknown")
	watermark_params   : dict with entropy_threshold/delta/proxy_window_size set to null
	source_meta        : subset of original midput record for traceability

Extra fields are ignored by analyze_outputs.py, so adding source_meta keeps
compatibility even when the output contains more entries than example.json.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from evaluate.base import DummyZScoreDetector, ProxyDetector
from evaluate.watermark_params import WatermarkParams
from concurrent.futures import ThreadPoolExecutor, as_completed


def _read_midput(path: str) -> Iterable[Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if isinstance(data, list):
		return data
	if isinstance(data, dict):
		# Single object; wrap for uniform handling
		return [data]
	raise ValueError(f"Unsupported JSON root type: {type(data)}")


def _sanitize_token(token: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]+", "-", token.strip()) or "item"
def build_output_stem(
	base_stem: str,
	model: Optional[str],
	params: WatermarkParams,
	algorithm: Optional[str] = None,
) -> str:
	def sanitize(s: Optional[str]) -> str:
		if not s:
			return "nomodel"
		return s.replace("/", "-").replace(":", "-").replace(" ", "-")

	m = sanitize(model)
	ent = params.entropy_threshold
	dlt = params.delta
	win = params.proxy_window_size
	ent_s = f"ent{ent}" if ent is not None else "entNA"
	dlt_s = f"d{dlt}" if dlt is not None else "dNA"
	win_s = f"w{win}" if win is not None else "wNA"
	alg = (algorithm or os.environ.get("WATERMARK_ALGORITHM", "unknown")).replace("/", "-").replace(":", "-").replace(" ", "-")
	return f"{base_stem}.{m}.{ent_s}.{dlt_s}.{win_s}.alg{alg}"


def save_result(output_dir: str, result: Dict[str, Any], stem: str) -> str:
	os.makedirs(output_dir, exist_ok=True)
	from datetime import datetime
	ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
	path = os.path.join(output_dir, f"{stem}.{ts}.json")
	with open(path, "w", encoding="utf-8") as f:
		json.dump(result, f, ensure_ascii=False, indent=2)
	return path


def _build_prompt(record: Dict[str, Any]) -> str:
	title = record.get("question_title") or record.get("title") or ""
	content = record.get("question_content") or record.get("prompt") or ""
	if title and content:
		return f"{title}\n\n{content}"
	return title or content or ""


def convert(
	input_path: str,
	output_dir: str,
	base_stem: Optional[str],
	attack: str,
	watermark_algorithm: Optional[str],
	entropy_threshold: Optional[float],
	delta: Optional[float],
	proxy_window_size: Optional[int],
	limit: Optional[int],
	detector: Optional[object],
	model: Optional[str],
	concurrency: int,
) -> int:
	records = list(_read_midput(input_path))
	os.makedirs(output_dir, exist_ok=True)

	stem_base = _sanitize_token(base_stem or os.path.splitext(os.path.basename(input_path))[0])

	out_count = 0
	global_index = 0
	for rec_idx, rec in enumerate(records):
		prompt_text = _build_prompt(rec)
		outputs = rec.get("output_list") or []
		if not isinstance(outputs, list):
			outputs = []

		# Respect limit by slicing outputs to process
		if limit is not None:
			if out_count >= limit:
				return out_count
			max_items = max(0, limit - out_count)
			texts_to_process = outputs[:max_items]
		else:
			texts_to_process = outputs

		# Build params once
		params_obj = WatermarkParams(
			entropy_threshold=entropy_threshold,
			delta=delta,
			proxy_window_size=proxy_window_size,
		)

		# Compute z-scores (parallel if requested)
		z_vals: List[Optional[float]] = [None] * len(texts_to_process)

		def _normalize_z(z_raw: Any) -> Optional[float]:
			try:
				if isinstance(z_raw, dict):
					v = z_raw.get("z_score")
					if v is None:
						for val in z_raw.values():
							if isinstance(val, (int, float)):
								v = val
								break
					return float(v) if v is not None else None
				return float(z_raw)
			except Exception:
				return None

		if detector is not None and len(texts_to_process) > 0:
			if concurrency and concurrency > 1:
				with ThreadPoolExecutor(max_workers=concurrency) as pool:
					futures = {pool.submit(detector.detect, t, params_obj): i for i, t in enumerate(texts_to_process) if isinstance(t, str) and t.strip()}
					for fut in as_completed(futures):
						i = futures[fut]
						try:
							z_raw = fut.result()
						except Exception:
							z_raw = None
						z_vals[i] = _normalize_z(z_raw)
			else:
				# Try batch_detect if available, else sequential
				try:
					if hasattr(detector, "batch_detect"):
						z_raws = detector.batch_detect(texts_to_process, params_obj)  # type: ignore
						for i, zr in enumerate(z_raws):
							z_vals[i] = _normalize_z(zr)
					else:
						for i, t in enumerate(texts_to_process):
							if isinstance(t, str) and t.strip():
								try:
									z_raw = detector.detect(t, params_obj)
								except Exception:
									z_raw = None
								z_vals[i] = _normalize_z(z_raw)
				except Exception:
					for i, t in enumerate(texts_to_process):
						if isinstance(t, str) and t.strip():
							try:
								z_raw = detector.detect(t, params_obj)
							except Exception:
								z_raw = None
							z_vals[i] = _normalize_z(z_raw)

		# Write outputs
		for out_idx, raw_text in enumerate(texts_to_process):
			z_val = z_vals[out_idx]

			entry = {
				"index": global_index,
				"prompt": prompt_text,
				"attack": attack,
				"sample_raw": raw_text,
				"sample_attacked": raw_text,
				"z_score": z_val,
				"watermark_algorithm": watermark_algorithm or "unknown",
				"watermark_params": {
					"entropy_threshold": entropy_threshold,
					"delta": delta,
					"proxy_window_size": proxy_window_size,
				},
				"model": model,
				"source_meta": {
					"record_index": rec_idx,
					"output_index": out_idx,
					"question_title": rec.get("question_title"),
					"question_id": rec.get("question_id"),
					"platform": rec.get("platform"),
					"difficulty": rec.get("difficulty"),
					"pass@1": rec.get("pass@1"),
					"metadata": rec.get("metadata"),
				},
			}

			stem_hint = _sanitize_token(rec.get("question_title") or "json")
			base_stem_item = f"{stem_base}.{stem_hint}.{out_idx:03d}"
			out_stem = build_output_stem(base_stem_item, model, params_obj, algorithm=watermark_algorithm)

			save_result(output_dir, entry, out_stem)
			out_count += 1
			global_index += 1

	return out_count


def init_detector(args: argparse.Namespace) -> Optional[object]:
	method = (args.detection_method or "proxy").lower()
	if method == "proxy":
		try:
			det = ProxyDetector(
				proxy_model=args.proxy_model or os.environ.get("WATERMARK_PROXY_MODEL"),
				entropy_threshold_default=args.entropy_threshold,
				secret_key=args.secret_key,
				window_size=args.proxy_window_size,
				z_threshold=args.z_threshold,
				prefix_str=args.proxy_template_prefix,
				suffix_str=args.proxy_template_suffix,
			)
			return det
		except Exception as e:
			print(f"[Warn] ProxyDetector init failed: {e}; falling back to DummyZScoreDetector.")
			return DummyZScoreDetector()
	else:
		# Default to dummy when method unrecognized or explicitly dummy
		return DummyZScoreDetector()


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert midput JSON to analyze_outputs-compatible samples.")
	parser.add_argument("--input", required=True, help="Path to midput JSON file")
	parser.add_argument("--output-dir", required=True, help="Directory to write per-sample JSON files")
	parser.add_argument("--base-stem", default=None, help="Base filename stem (default: derived from input file name)")
	parser.add_argument("--attack", default="none", help="Attack label to write into outputs")
	parser.add_argument(
		"--watermark-algorithm",
		dest="watermark_algorithm",
		default=os.environ.get("WATERMARK_ALGORITHM", None),
		help="Watermark algorithm label for outputs",
	)
	parser.add_argument(
		"--alg",
		dest="watermark_algorithm",
		help="Alias for --watermark-algorithm",
	)
	parser.add_argument(
		"--entropy-threshold",
		type=float,
		default=None,
		help="Entropy threshold to record in watermark_params",
	)
	parser.add_argument(
		"--delta",
		type=float,
		default=None,
		help="Delta to record in watermark_params",
	)
	parser.add_argument(
		"--proxy-window-size",
		type=int,
		default=None,
		help="Proxy window size to record in watermark_params",
	)
	parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of outputs to emit")
	parser.add_argument("--model", type=str, default=None, help="Model name used for naming consistency")
	parser.add_argument(
		"--detection-method",
		dest="detection_method",
		choices=["proxy", "dummy"],
		default=os.environ.get("WATERMARK_ALGORITHM", "proxy"),
		help="Detector to use for z-score generation",
	)
	parser.add_argument("--proxy-model", type=str, default=os.environ.get("WATERMARK_PROXY_MODEL"))
	parser.add_argument("--secret-key", type=int, default=os.environ.get("WATERMARK_SECRET_KEY"))
	parser.add_argument("--z-threshold", type=float, default=4.0)
	parser.add_argument(
		"--proxy-template-prefix",
		type=str,
		default=os.environ.get("WATERMARK_PROXY_TEMPLATE_PREFIX", "<|fim_prefix|>"),
	)
	parser.add_argument(
		"--proxy-template-suffix",
		type=str,
		default=os.environ.get("WATERMARK_PROXY_TEMPLATE_SUFFIX", "<|fim_suffix|>\n<|fim_middle|>"),
	)
	parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "1")))

	args = parser.parse_args()

	det = init_detector(args)

	written = convert(
		input_path=args.input,
		output_dir=args.output_dir,
		base_stem=args.base_stem,
		attack=args.attack,
		watermark_algorithm=args.watermark_algorithm,
		entropy_threshold=args.entropy_threshold,
		delta=args.delta,
		proxy_window_size=args.proxy_window_size,
		limit=args.limit,
		detector=det,
		model=args.model,
		concurrency=args.concurrency,
	)
	print(f"Wrote {written} output file(s) to {args.output_dir}")


if __name__ == "__main__":
	main()
