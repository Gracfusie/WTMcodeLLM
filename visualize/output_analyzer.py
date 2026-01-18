from __future__ import annotations

# import argparse
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
        rows.append(
            dict(
                # path=p,
                # filename=os.path.basename(p),
                num_total_tokens=data.get('num_total_tokens'),
                num_green_tokens=data.get('num_green_tokens'),
                num_tokens_scored=data.get('num_tokens_scored'),
                green_fraction=data.get('green_fraction'),
                z_score=data.get('z_score'),
            )
        )

    return pd.DataFrame(rows)

def plot_z_distributions(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    zs = df['z_score'].dropna().values
    if zs.size == 0:
        print('No z_score values to plot')
        return
    bins = np.histogram_bin_edges(zs, bins='fd')
    plt.figure(figsize=(8, 5))
    plt.hist(zs, bins=bins, density=True, alpha=0.8, edgecolor='none')
    plt.title('z-score distribution')
    plt.xlabel('z-score')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'zdist_all.png')
    plt.savefig(out_path)
    plt.close()

def plot_tail_ratios(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    zs = df['z_score'].dropna().values
    if zs.size == 0:
        print('No z_score values to plot tail ratios')
        return
    zmin, zmax = np.nanmin(zs), np.nanmax(zs)
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        print('Invalid z_score range')
        return
    zmax_capped = min(zmax, 15.0)
    t_start = max(-3.0, min(zmin, 15.0))
    thresholds = np.linspace(t_start, zmax_capped, 101)
    ratios = [(zs > t).mean() for t in thresholds]
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, ratios)
    plt.title('Tail ratio: proportion(z > threshold)')
    plt.xlabel('z-threshold')
    plt.ylabel('proportion(z > threshold)')
    plt.xlim(left=t_start, right=15)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tail_all.png')
    plt.savefig(out_path)
    plt.close()

def plot_len_vs_z(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sub = df[['num_tokens_scored', 'z_score']].dropna()
    if sub.empty:
        print('No data for length vs z-score scatter')
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(sub['num_tokens_scored'], sub['z_score'], s=16, alpha=0.6)
    ax = plt.gca()
    ax.axhline(4, color='red', linestyle='--', linewidth=1)
    ax.axvline(200, color='blue', linestyle='--', linewidth=1)
    # Label horizontal line at a small left margin, using y-axis transform (y in data coords)
    ax.text(0.02, 4, '4', color='red', va='bottom', ha='left', transform=ax.get_yaxis_transform())
    # Label vertical line near bottom margin, using x-axis transform (x in data coords)
    ax.text(200, 0.02, '200', color='blue', va='bottom', ha='left', rotation=90, transform=ax.get_xaxis_transform())
    plt.title('Token length vs z-score')
    plt.xlabel('num_tokens_scored')
    plt.ylabel('z-score')
    plt.xscale('log')
    plt.yscale('log')
    # plt.tight_layout()
    out_path = os.path.join(out_dir, 'scatter_len_score_vs_z.png')
    plt.savefig(out_path)
    plt.close()


def plot_tlen_vs_z(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sub = df[['num_total_tokens', 'z_score']].dropna()
    if sub.empty:
        print('No data for total length vs z-score scatter')
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(sub['num_total_tokens'], sub['z_score'], s=16, alpha=0.6)
    ax = plt.gca()
    # ax.axhline(4, color='red', linestyle='--', linewidth=1)
    # ax.axvline(200, color='blue', linestyle='--', linewidth=1)
    # ax.text(0.02, 4, '4', color='red', va='bottom', ha='left', transform=ax.get_yaxis_transform())
    # ax.text(200, 0.02, '200', color='blue', va='bottom', ha='left', rotation=90, transform=ax.get_xaxis_transform())
    plt.title('Total token length vs z-score')
    plt.xlabel('num_total_tokens')
    plt.ylabel('z-score')
    plt.xscale('log')
    plt.yscale('log')
    out_path = os.path.join(out_dir, 'scatter_tlen_vs_z.png')
    plt.savefig(out_path)
    plt.close()


def plot_tlen_vs_len(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sub = df[['num_total_tokens', 'num_tokens_scored']].dropna()
    if sub.empty:
        print('No data for total length vs scored length scatter')
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(sub['num_total_tokens'], sub['num_tokens_scored'], s=16, alpha=0.6)
    ax = plt.gca()
    lim_max = max(sub['num_total_tokens'].max(), sub['num_tokens_scored'].max())
    lim_min = min(sub['num_total_tokens'].min(), sub['num_tokens_scored'].min())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='gray', linewidth=1)
    plt.title('Total tokens vs scored tokens')
    plt.xlabel('num_total_tokens')
    plt.ylabel('num_tokens_scored')
    plt.xscale('log')
    plt.yscale('log')
    out_path = os.path.join(out_dir, 'scatter_tlen_vs_len.png')
    plt.savefig(out_path)
    plt.close()

def main():
    out_dir = 'output/proxy-format'
    figs_dir = os.path.join(out_dir, 'analysis')

    df = collect_results(out_dir)
    assert not df.empty, "No results found!"

    out_csv = os.path.join(out_dir, 'analysis', 'all_results_simple.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print('Aggregated simple results saved to', out_csv)

    plot_z_distributions(df, figs_dir)

    plot_tail_ratios(df, figs_dir)

    plot_len_vs_z(df, figs_dir)

    plot_tlen_vs_z(df, figs_dir)

    plot_tlen_vs_len(df, figs_dir)

    full = True
    if full:

        neg_sample_dir = 'output/non-proxy0.5-n'

        # Compute detection metrics vs negatives
        try:
            df_neg = collect_results(neg_sample_dir)
            z_pos = df['z_score'].dropna().values
            z_neg = df_neg['z_score'].dropna().values if not df_neg.empty else np.array([])
            if z_pos.size == 0 or z_neg.size == 0:
                print('Insufficient data for detection metrics (pos/neg).')
                return

            # Threshold metrics at z=4
            threshold = 4.0
            tpr = float((z_pos > threshold).mean())
            fpr = float((z_neg > threshold).mean())

            # ROC and AUROC over shared thresholds
            zmin = float(np.nanmin(np.concatenate([z_pos, z_neg])))
            zmax = float(np.nanmax(np.concatenate([z_pos, z_neg])))
            zmax_capped = min(zmax, 15.0)
            t_start = max(-3.0, min(zmin, 15.0))
            thresholds = np.linspace(t_start, zmax_capped, 301)
            tprs = [(z_pos > t).mean() for t in thresholds]
            fprs = [(z_neg > t).mean() for t in thresholds]

            # AUROC via trapezoidal rule over FPR-TPR curve
            # Ensure arrays sorted by FPR ascending
            order = np.argsort(fprs)
            fprs_sorted = np.array(fprs)[order]
            tprs_sorted = np.array(tprs)[order]
            auroc = float(np.trapz(tprs_sorted, fprs_sorted))

            # Save metrics
            os.makedirs(figs_dir, exist_ok=True)
            metrics_txt = os.path.join(figs_dir, 'detection_metrics.txt')
            with open(metrics_txt, 'w') as f:
                f.write(f"threshold={threshold}\n")
                f.write(f"TPR(pos, z>thr)={tpr:.4f}\n")
                f.write(f"FPR(neg, z>thr)={fpr:.4f}\n")
                f.write(f"AUROC={auroc:.6f}\n")
            print('Detection metrics saved to', metrics_txt)

            # Plot ROC
            plt.figure(figsize=(6, 6))
            plt.plot(fprs_sorted, tprs_sorted, drawstyle='steps-post', linewidth=1)
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'ROC (AUROC={auroc:.3f})')
            roc_path = os.path.join(figs_dir, 'roc_simple.png')
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()
        except Exception as e:
            print('Error computing detection metrics:', e)

if __name__ == '__main__':
    main()

