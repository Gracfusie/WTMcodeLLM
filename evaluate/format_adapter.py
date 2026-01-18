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
	watermark_algorithm: str,
	limit: Optional[int],
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

		for out_idx, raw_text in enumerate(outputs):
			if limit is not None and out_count >= limit:
				return out_count

			entry = {
				"index": global_index,
				"prompt": prompt_text,
				"attack": attack,
				"sample_raw": raw_text,
				"sample_attacked": raw_text,
				"z_score": None,
				"watermark_algorithm": watermark_algorithm,
				"watermark_params": {
					"entropy_threshold": None,
					"delta": None,
					"proxy_window_size": None,
				},
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

			filename = f"{stem_base}_{global_index:05d}.json"
			out_path = os.path.join(output_dir, filename)
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump(entry, f, ensure_ascii=False, indent=2)

			out_count += 1
			global_index += 1

	return out_count


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert midput JSON to analyze_outputs-compatible samples.")
	parser.add_argument("--input", required=True, help="Path to midput JSON file")
	parser.add_argument("--output-dir", required=True, help="Directory to write per-sample JSON files")
	parser.add_argument("--base-stem", default=None, help="Base filename stem (default: derived from input file name)")
	parser.add_argument("--attack", default="none", help="Attack label to write into outputs")
	parser.add_argument(
		"--watermark-algorithm",
		default=os.environ.get("WATERMARK_ALGORITHM", "unknown"),
		help="Watermark algorithm label for outputs",
	)
	parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of outputs to emit")

	args = parser.parse_args()

	written = convert(
		input_path=args.input,
		output_dir=args.output_dir,
		base_stem=args.base_stem,
		attack=args.attack,
		watermark_algorithm=args.watermark_algorithm,
		limit=args.limit,
	)
	print(f"Wrote {written} output file(s) to {args.output_dir}")


if __name__ == "__main__":
	main()
