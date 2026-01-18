#!/usr/bin/env python3
"""
Collect and analyze JSON results under evaluate/output.

Features:
- Load all JSON files in `output_dir` and validate they contain expected fields.
- Aggregate into a pandas DataFrame with columns for algorithm and watermark params.
- Plot z-score distributions per (algorithm + params).
- Compute TPR/FPR vs z-threshold using files with algorithm=="none" as negatives.

Usage:
    python analyze_outputs.py --output-dir output --out-figs-dir output/analysis

"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_json_files(output_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(output_dir):
        for fn in filenames:
            if fn.endswith('.json'):
                files.append(os.path.join(root, fn))
    return sorted(files)


def load_result(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def parse_filename_for_alg(fname: str) -> Optional[str]:
    # look for .alg<name> suffix before extension
    m = re.search(r"\.alg([A-Za-z0-9_\-]+)\.json$", fname)
    if m:
        return m.group(1)
    return None


def normalize_params(p: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    ent = p.get('entropy_threshold') if p else None
    dlt = p.get('delta') if p else None
    win = p.get('proxy_window_size') if p else None
    return (ent, dlt, win)


def collect_results(output_dir: str) -> pd.DataFrame:
    files = find_json_files(output_dir)
    rows = []
    for p in files:
        data = load_result(p)
        if data is None:
            continue
        fname = os.path.basename(p)
        alg = data.get('watermark_algorithm') if isinstance(data, dict) else None
        if not alg:
            alg = parse_filename_for_alg(fname) or 'unknown'

        params = data.get('watermark_params', {}) if isinstance(data, dict) else {}
        ent, dlt, win = normalize_params(params)
        z = None
        try:
            z = float(data.get('z_score')) if data.get('z_score') is not None else None
        except Exception:
            z = None

        model = data.get('model') if isinstance(data, dict) else None
        row = dict(
            path=p,
            filename=fname,
            algorithm=alg,
            entropy_threshold=ent,
            delta=dlt,
            proxy_window_size=win,
            z_score=z,
            model=model,
            raw=data.get('sample_raw') if isinstance(data, dict) else None,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_z_distributions(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # group by algorithm + params
    groups = df.groupby(['algorithm', 'entropy_threshold', 'delta', 'proxy_window_size'])
    for (alg, ent, dlt, win), g in groups:
        zs = g['z_score'].dropna().values
        if zs.size == 0:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(zs, kde=True, stat='density')
        plt.title(f"z-score dist: {alg} ent={ent} d={dlt} w={win}")
        plt.xlabel('z-score')
        out_path = os.path.join(out_dir, f"zdist_{alg}_ent{ent}_d{dlt}_w{win}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    # Overlay all params in one figure for quick comparison
    overlay = df.dropna(subset=['z_score']).copy()
    if not overlay.empty:
        overlay['label'] = overlay.apply(
            lambda r: f"{r['algorithm']}|ent={r['entropy_threshold']}|d={r['delta']}|w={r['proxy_window_size']}",
            axis=1,
        )
        counts = overlay['label'].value_counts().to_dict()
        overlay['label_with_count'] = overlay['label'].apply(lambda l: f"{l} (n={counts.get(l, 0)})")
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=overlay, x='z_score', hue='label_with_count', common_norm=False)
        plt.title('z-score distributions (all params)')
        plt.xlabel('z-score')
        plt.tight_layout()
        out_path = os.path.join(out_dir, 'zdist_all_params.png')
        plt.savefig(out_path)
        plt.close()


def _mask_eq(series: pd.Series, target: float) -> pd.Series:
    try:
        return series.notna() & series.astype(float).eq(float(target))
    except Exception:
        return pd.Series([False] * len(series), index=series.index)


def plot_proxy_param_sweeps(
    df: pd.DataFrame,
    out_dir: str,
    base_ent: float = 0.5,
    base_delta: float = 2.0,
    base_win: int = 256,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    proxy = df[(df['algorithm'] == 'proxy') & df['z_score'].notna()].copy()
    if proxy.empty:
        return

    def _make_plot(sub: pd.DataFrame, label_field: str, title: str, fname: str) -> None:
        if sub.empty:
            return
        counts = sub[label_field].value_counts().to_dict()
        sub['label_with_count'] = sub[label_field].apply(lambda l: f"{l} (n={counts.get(l, 0)})")
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=sub, x='z_score', hue='label_with_count', common_norm=False)
        plt.title(title)
        plt.xlabel('z-score')
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path)
        plt.close()

    # Sweep entropy_threshold with delta/window fixed at baseline
    ent_mask = _mask_eq(proxy['delta'], base_delta) & _mask_eq(proxy['proxy_window_size'], base_win)
    ent_df = proxy[ent_mask].copy()
    if not ent_df.empty:
        ent_df['label'] = ent_df['entropy_threshold'].apply(lambda v: f"ent={v}")
        _make_plot(ent_df, 'label', f'proxy z-score by entropy (delta={base_delta}, win={base_win})', 'proxy_zdist_sweep_ent.png')

    # Sweep delta with entropy/window fixed at baseline
    delta_mask = _mask_eq(proxy['entropy_threshold'], base_ent) & _mask_eq(proxy['proxy_window_size'], base_win)
    delta_df = proxy[delta_mask].copy()
    if not delta_df.empty:
        delta_df['label'] = delta_df['delta'].apply(lambda v: f"delta={v}")
        _make_plot(delta_df, 'label', f'proxy z-score by delta (ent={base_ent}, win={base_win})', 'proxy_zdist_sweep_delta.png')

    # Sweep proxy_window_size with entropy/delta fixed at baseline
    win_mask = _mask_eq(proxy['entropy_threshold'], base_ent) & _mask_eq(proxy['delta'], base_delta)
    win_df = proxy[win_mask].copy()
    if not win_df.empty:
        win_df['label'] = win_df['proxy_window_size'].apply(lambda v: f"win={int(v)}")
        _make_plot(win_df, 'label', f'proxy z-score by window (ent={base_ent}, delta={base_delta})', 'proxy_zdist_sweep_win.png')


def compute_tpr_fpr(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # negatives: algorithm == 'none'
    negs = df[df['algorithm'] == 'none']
    if negs.empty:
        print("No 'none' algorithm negative samples found; cannot compute TPR/FPR baseline.")
        return

    # For each positive group (algorithm != none), compute ROC-like curve against negs
    pos_groups = df[df['algorithm'] != 'none'].groupby(['algorithm', 'entropy_threshold', 'delta', 'proxy_window_size'])
    neg_z = negs['z_score'].dropna().values
    for (alg, ent, dlt, win), pos in pos_groups:
        pos_z = pos['z_score'].dropna().values
        if pos_z.size == 0:
            continue
        all_z = np.concatenate([pos_z, neg_z])
        labels = np.concatenate([np.ones(len(pos_z)), np.zeros(len(neg_z))])
        thresholds = np.linspace(np.nanmin(all_z), np.nanmax(all_z), 101)
        tprs = []
        fprs = []
        for t in thresholds:
            preds = (all_z > t).astype(int)
            tp = int(((preds == 1) & (labels == 1)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            tn = int(((preds == 0) & (labels == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tprs.append(tpr)
            fprs.append(fpr)

        # Save ROC curve plot
        plt.figure(figsize=(6, 6))
        plt.plot(fprs, tprs, marker='.')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC: {alg} ent={ent} d={dlt} w={win}')
        plt.grid(True)
        roc_path = os.path.join(out_dir, f"roc_{alg}_ent{ent}_d{dlt}_w{win}.png")
        plt.savefig(roc_path)
        plt.close()

        # Save TPR/FPR vs threshold
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(thresholds, tprs, label='TPR')
        ax.plot(thresholds, fprs, label='FPR')
        ax.set_xlabel('z-threshold')
        ax.set_ylabel('Rate')
        ax.set_title(f'TPR/FPR vs z-threshold: {alg} ent={ent} d={dlt} w={win}')
        ax.legend()
        thr_path = os.path.join(out_dir, f"tprfpr_{alg}_ent{ent}_d{dlt}_w{win}.png")
        fig.tight_layout()
        fig.savefig(thr_path)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--out-figs-dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir
    figs_dir = args.out_figs_dir or os.path.join(out_dir, 'analysis')

    df = collect_results(out_dir)
    if df.empty:
        print('No result JSON files found under', out_dir)
        return

    # confirm filename info present in JSON
    missing = []
    for idx, row in df.iterrows():
        if pd.isna(row['z_score']):
            missing.append(row['path'])
    if missing:
        print(f"Warning: {len(missing)} files missing z_score. Examples: {missing[:5]}")

    # Save aggregated CSV
    agg_csv = os.path.join(figs_dir, 'all_results.csv')
    os.makedirs(figs_dir, exist_ok=True)
    df.to_csv(agg_csv, index=False)
    print('Aggregated results saved to', agg_csv)

    # 1. z-score distributions
    plot_z_distributions(df, figs_dir)

    # 1b. proxy param sweeps (vary one param, fix others to defaults)
    plot_proxy_param_sweeps(df, figs_dir)

    # 2. TPR/FPR analysis vs z-threshold
    compute_tpr_fpr(df, figs_dir)


if __name__ == '__main__':
    main()
