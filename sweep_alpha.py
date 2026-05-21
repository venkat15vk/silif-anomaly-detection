"""
Alpha sweep for SilIF (silhouette weight in the final risk score).

SilIF combines:
    final = base_IF_score + alpha * (1 - silhouette)

This script runs SilIF at several alpha values across multiple seeds and
reports which alpha gives the strongest result. Also re-runs IsolationForest
and GlobalKMeans on the same seeds as references.

Usage:
    python3 sweep_alpha.py                      # full sweep on full dataset
    python3 sweep_alpha.py --sample 100000      # faster, subsampled
    python3 sweep_alpha.py --seeds 3            # fewer seeds

Outputs results/alpha_sweep.csv and a summary table on console.
"""

from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    load_data,
    silif_method,
    baseline_isolation_forest,
    baseline_global_kmeans,
    evaluate,
)

ALPHA_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def run_one_seed(df: pd.DataFrame, y: np.ndarray, seed: int,
                 alphas: List[float]) -> Dict[str, Dict[str, float]]:
    """Run SilIF at each alpha + the two reference baselines, once with given seed."""
    results = {}
    print(f"\n[seed {seed}] -----------------------------")

    for a in alphas:
        t0 = time.time()
        s = silif_method(df, seed, alpha=a)
        m = evaluate(s, y)
        results[f"SilIF(a={a})"] = m
        print(f"  SilIF(a={a:<4}): {time.time()-t0:.1f}s  "
              f"AUC_PR={m['AUC_PR']:.4f}  AUC_ROC={m['AUC_ROC']:.4f}")

    t0 = time.time()
    results["IF"] = evaluate(baseline_isolation_forest(df, seed), y)
    print(f"  IsolationForest: {time.time()-t0:.1f}s  "
          f"AUC_PR={results['IF']['AUC_PR']:.4f}")

    t0 = time.time()
    results["GlobalKMeans"] = evaluate(baseline_global_kmeans(df, seed), y)
    print(f"  GlobalKMeans:    {time.time()-t0:.1f}s  "
          f"AUC_PR={results['GlobalKMeans']['AUC_PR']:.4f}")

    return results


def summarize(all_runs: List[Dict[str, Dict[str, float]]], metric: str) -> pd.DataFrame:
    methods = list(all_runs[0].keys())
    rows = []
    for m in methods:
        vals = np.array([r[m][metric] for r in all_runs])
        rows.append({
            "method": m,
            f"{metric}_mean": vals.mean(),
            f"{metric}_std":  vals.std(ddof=1) if len(vals) > 1 else 0.0,
            f"{metric}_min":  vals.min(),
            f"{metric}_max":  vals.max(),
        })
    return pd.DataFrame(rows)


def paired_compare(all_runs, m_a: str, m_b: str, metric: str = "AUC_PR"):
    from scipy import stats
    a = np.array([r[m_a][metric] for r in all_runs])
    b = np.array([r[m_b][metric] for r in all_runs])
    diff = a - b
    if len(diff) < 2:
        return {"mean_diff": float(diff[0]) if len(diff) else 0.0,
                "wins": int((diff > 0).sum()), "n": len(diff),
                "p": float("nan")}
    t, p = stats.ttest_rel(a, b)
    return {"mean_diff": float(diff.mean()),
            "wins": int((diff > 0).sum()), "n": len(diff),
            "p": float(p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--min-tx", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHA_VALUES,
                        help="Alpha values to sweep")
    args = parser.parse_args()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print(f"[sweep] Loading data (sample={args.sample}, min_tx={args.min_tx}) ...")
    df, kind = load_data(sample=args.sample, seed=args.base_seed, min_tx=args.min_tx)
    y = df["isFraud"].values.astype(int)
    print(f"[sweep] Data: {len(df):,} txns, {y.sum():,} fraud "
          f"({100*y.mean():.3f}%), dataset={kind}")
    print(f"[sweep] Alphas: {args.alphas}")
    print(f"[sweep] Seeds:  {args.seeds}")

    seeds = [args.base_seed + i for i in range(args.seeds)]

    t_start = time.time()
    all_runs = []
    for seed in seeds:
        all_runs.append(run_one_seed(df, y, seed, args.alphas))
    print(f"\n[sweep] Total time: {(time.time()-t_start)/60:.1f} min")

    # Summarize
    print("\n=== ALPHA SWEEP: AUC_PR (mean +/- std across seeds) ===")
    summ_pr = summarize(all_runs, "AUC_PR")
    print(summ_pr.round(4).to_string(index=False))

    print("\n=== ALPHA SWEEP: AUC_ROC (mean +/- std across seeds) ===")
    summ_roc = summarize(all_runs, "AUC_ROC")
    print(summ_roc.round(4).to_string(index=False))

    # Find best alpha by AUC_PR
    silif_rows = summ_pr[summ_pr["method"].str.startswith("SilIF(")]
    best_idx = silif_rows["AUC_PR_mean"].idxmax()
    best_method = silif_rows.loc[best_idx, "method"]
    best_mean = silif_rows.loc[best_idx, "AUC_PR_mean"]
    print(f"\n[sweep] Best alpha by AUC_PR: {best_method} (mean={best_mean:.4f})")

    # Head-to-head: best SilIF vs IF and GlobalKMeans
    print(f"\n=== {best_method} vs reference baselines (AUC_PR) ===")
    for b in ["IF", "GlobalKMeans"]:
        c = paired_compare(all_runs, best_method, b, "AUC_PR")
        verdict = "WIN" if c["wins"] >= 0.8 * c["n"] and c["p"] < 0.10 else \
                  "consistent" if c["wins"] > c["n"]/2 else "weak"
        print(f"  vs {b:<14} diff={c['mean_diff']:+.5f}  "
              f"wins={c['wins']}/{c['n']}  p={c['p']:.4f}  [{verdict}]")

    print(f"\n=== {best_method} vs reference baselines (AUC_ROC) ===")
    for b in ["IF", "GlobalKMeans"]:
        c = paired_compare(all_runs, best_method, b, "AUC_ROC")
        verdict = "WIN" if c["wins"] >= 0.8 * c["n"] and c["p"] < 0.10 else \
                  "consistent" if c["wins"] > c["n"]/2 else "weak"
        print(f"  vs {b:<14} diff={c['mean_diff']:+.5f}  "
              f"wins={c['wins']}/{c['n']}  p={c['p']:.4f}  [{verdict}]")

    # Save per-seed CSV
    rows = []
    for seed, run in zip(seeds, all_runs):
        for method, metrics in run.items():
            row = {"seed": seed, "method": method}
            row.update(metrics)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "alpha_sweep.csv", index=False)
    summ_pr.to_csv(out_dir / "alpha_sweep_summary_aucpr.csv", index=False)
    summ_roc.to_csv(out_dir / "alpha_sweep_summary_aucroc.csv", index=False)
    print(f"\n[out] Wrote: {out_dir / 'alpha_sweep.csv'}")


if __name__ == "__main__":
    main()
