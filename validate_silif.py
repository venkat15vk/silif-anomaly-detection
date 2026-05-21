"""
Multi-seed validation: SilIF vs Isolation Forest (and other top baselines).

Runs each method on the same data with multiple random seeds, then reports:
  - Per-seed scores
  - Mean +/- std across seeds
  - Paired t-test SilIF vs each baseline (does SilIF significantly beat it?)
  - Win rate (on how many seeds did SilIF beat the baseline?)

Usage:
    python3 validate_silif.py                  # 5 seeds, full dataset
    python3 validate_silif.py --seeds 3        # 3 seeds (faster)
    python3 validate_silif.py --sample 100000  # subsample for speed

Output: results/multiseed_results.csv and console summary.
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

# Import everything we need from the main script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    load_data,
    per_transaction_features,
    silif_method,
    baseline_isolation_forest,
    baseline_hbos,
    baseline_ecod,
    baseline_global_kmeans,
    evaluate,
)


def run_one_seed(df: pd.DataFrame, y: np.ndarray, seed: int) -> Dict[str, Dict[str, float]]:
    """Run SilIF + top baselines once with a given seed."""
    results = {}
    print(f"\n[seed {seed}] -----------------------------")

    t0 = time.time()
    s_silif = silif_method(df, seed)
    results["SilIF"] = evaluate(s_silif, y)
    print(f"  SilIF:           {time.time()-t0:.1f}s  AUC_PR={results['SilIF']['AUC_PR']:.4f}")

    t0 = time.time()
    s_if = baseline_isolation_forest(df, seed)
    results["IF"] = evaluate(s_if, y)
    print(f"  IsolationForest: {time.time()-t0:.1f}s  AUC_PR={results['IF']['AUC_PR']:.4f}")

    t0 = time.time()
    s_gkm = baseline_global_kmeans(df, seed)
    results["GlobalKMeans"] = evaluate(s_gkm, y)
    print(f"  GlobalKMeans:    {time.time()-t0:.1f}s  AUC_PR={results['GlobalKMeans']['AUC_PR']:.4f}")

    t0 = time.time()
    s_hbos = baseline_hbos(df, seed)
    results["HBOS"] = evaluate(s_hbos, y)
    print(f"  HBOS:            {time.time()-t0:.1f}s  AUC_PR={results['HBOS']['AUC_PR']:.4f}")

    t0 = time.time()
    s_ecod = baseline_ecod(df, seed)
    results["ECOD"] = evaluate(s_ecod, y)
    print(f"  ECOD:            {time.time()-t0:.1f}s  AUC_PR={results['ECOD']['AUC_PR']:.4f}")

    return results


def summarize(all_runs: List[Dict[str, Dict[str, float]]], metric: str = "AUC_PR") -> pd.DataFrame:
    """Aggregate per-seed results into mean +/- std per method."""
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
            "n_seeds": len(vals),
        })
    return pd.DataFrame(rows)


def paired_comparison(all_runs: List[Dict[str, Dict[str, float]]],
                      method: str, baseline: str,
                      metric: str = "AUC_PR") -> Dict[str, float]:
    """Paired t-test: does `method` consistently beat `baseline`?"""
    from scipy import stats
    a = np.array([r[method][metric]   for r in all_runs])
    b = np.array([r[baseline][metric] for r in all_runs])
    diffs = a - b
    wins = int((diffs > 0).sum())
    if len(diffs) < 2:
        return {
            "mean_diff": float(diffs[0]) if len(diffs) else 0.0,
            "wins": wins, "n": len(diffs),
            "t_stat": float("nan"), "p_value": float("nan"),
        }
    t_stat, p_value = stats.ttest_rel(a, b)
    return {
        "mean_diff": float(diffs.mean()),
        "wins": wins, "n": len(diffs),
        "t_stat": float(t_stat), "p_value": float(p_value),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds to run (default 5)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Subsample size (0 = full dataset)")
    parser.add_argument("--min-tx", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42,
                        help="First seed; subsequent seeds are base+1, base+2, ...")
    args = parser.parse_args()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # Load data ONCE — same data for all seeds (only the methods are reseeded)
    print(f"[multiseed] Loading data (sample={args.sample}, min_tx={args.min_tx}) ...")
    df, kind = load_data(sample=args.sample, seed=args.base_seed, min_tx=args.min_tx)
    y = df["isFraud"].values.astype(int)
    print(f"[multiseed] Data: {len(df):,} txns, {y.sum():,} fraud "
          f"({100*y.mean():.3f}%), dataset={kind}")

    seeds = [args.base_seed + i for i in range(args.seeds)]
    print(f"[multiseed] Running {args.seeds} seeds: {seeds}")

    t_start = time.time()
    all_runs = []
    for seed in seeds:
        all_runs.append(run_one_seed(df, y, seed))
    total_t = time.time() - t_start
    print(f"\n[multiseed] Total time: {total_t/60:.1f} min")

    # Summarize
    print("\n=== MEAN +/- STD ACROSS SEEDS (metric: AUC_PR) ===")
    summ_pr  = summarize(all_runs, "AUC_PR")
    print(summ_pr.round(4).to_string(index=False))

    print("\n=== MEAN +/- STD ACROSS SEEDS (metric: AUC_ROC) ===")
    summ_roc = summarize(all_runs, "AUC_ROC")
    print(summ_roc.round(4).to_string(index=False))

    # Paired comparisons: SilIF vs each baseline
    print("\n=== PAIRED COMPARISONS: SilIF vs baseline (metric: AUC_PR) ===")
    print(f"{'baseline':<15} {'mean_diff':>11} {'wins/n':>10} {'t':>8} {'p_value':>10}")
    for b in ["IF", "GlobalKMeans", "HBOS", "ECOD"]:
        c = paired_comparison(all_runs, "SilIF", b, "AUC_PR")
        print(f"{b:<15} {c['mean_diff']:>+11.5f} {c['wins']:>4}/{c['n']:<4} "
              f"{c['t_stat']:>8.3f} {c['p_value']:>10.4f}")

    print("\n=== PAIRED COMPARISONS: SilIF vs baseline (metric: AUC_ROC) ===")
    print(f"{'baseline':<15} {'mean_diff':>11} {'wins/n':>10} {'t':>8} {'p_value':>10}")
    for b in ["IF", "GlobalKMeans", "HBOS", "ECOD"]:
        c = paired_comparison(all_runs, "SilIF", b, "AUC_ROC")
        print(f"{b:<15} {c['mean_diff']:>+11.5f} {c['wins']:>4}/{c['n']:<4} "
              f"{c['t_stat']:>8.3f} {c['p_value']:>10.4f}")

    # Save raw per-seed results
    rows = []
    for seed, run in zip(seeds, all_runs):
        for method, metrics in run.items():
            row = {"seed": seed, "method": method}
            row.update(metrics)
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "multiseed_results.csv", index=False)
    summ_pr.to_csv(out_dir / "multiseed_summary_aucpr.csv", index=False)
    summ_roc.to_csv(out_dir / "multiseed_summary_aucroc.csv", index=False)
    print(f"\n[out] Wrote per-seed results: {out_dir / 'multiseed_results.csv'}")
    print(f"[out] Wrote summaries:        {out_dir / 'multiseed_summary_*.csv'}")

    # Final verdict
    print("\n=== VERDICT ===")
    silif_mean = summ_pr.loc[summ_pr["method"] == "SilIF", "AUC_PR_mean"].iloc[0]
    if_mean    = summ_pr.loc[summ_pr["method"] == "IF",    "AUC_PR_mean"].iloc[0]
    c = paired_comparison(all_runs, "SilIF", "IF", "AUC_PR")
    if c["wins"] >= 0.8 * c["n"] and c["p_value"] < 0.10:
        print(f"  SilIF beats IF on {c['wins']}/{c['n']} seeds, "
              f"p={c['p_value']:.4f}, mean diff +{c['mean_diff']:.5f} AUC_PR. "
              f"REAL EFFECT.")
    elif c["wins"] > c["n"] / 2:
        print(f"  SilIF beats IF on {c['wins']}/{c['n']} seeds, "
              f"p={c['p_value']:.4f}, mean diff +{c['mean_diff']:.5f} AUC_PR. "
              f"Consistent but weak.")
    else:
        print(f"  SilIF wins only {c['wins']}/{c['n']} seeds, "
              f"p={c['p_value']:.4f}. Effect NOT robust.")


if __name__ == "__main__":
    main()
