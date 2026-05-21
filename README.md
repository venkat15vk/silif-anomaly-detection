# SilIF — Silhouette-Augmented Isolation Forest

Reproducibility code for the paper:

> **SilIF: Silhouette-Augmented Isolation Forest for Unsupervised Transaction Fraud Detection**
> Venkatakrishnan Gopalakrishnan, 2026.
> arXiv preprint: [link to be added after submission]

## What this is

SilIF augments the standard Isolation Forest with a silhouette-based scoring layer. For each point, the method extracts a per-tree path-length "fingerprint", clusters fingerprints into K structural groups via K-means, and computes a silhouette score that measures how well each point fits its assigned group. This signal is combined with the base IF anomaly score via a single weight α. The α = 0 case recovers plain Isolation Forest.

On IEEE-CIS Fraud Detection (~590K transactions, 3.5% fraud), SilIF with α = 1.0 improves Isolation Forest's AUC-PR by +0.008 (5/5 random seeds, paired t-test p = 0.046). On Sparkov (synthetic, 1.85M transactions, 0.52% fraud), the silhouette augmentation does not help. The paper characterizes when this approach is and is not effective.

## What's in this repo

- `run_experiments.py` — main experiment runner. Auto-detects PaySim, IEEE-CIS, or Sparkov from CSV columns in the working directory. Runs SilIF + all unsupervised baselines (Isolation Forest, HBOS, ECOD, Global K-Means, LOF, kNN-distance). Writes results to `results/`.
- `validate_silif.py` — multi-seed validation harness. Runs each method across multiple seeds, reports mean ± std, paired t-tests vs each baseline, and a verdict line.
- `sweep_alpha.py` — hyperparameter sweep over α ∈ {0, 0.25, 0.5, 1.0, 2.0, 4.0} across multiple seeds.

## Requirements

- Python 3.9 or newer
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`
- Optional: `pyod` (for the CBLOF baseline; if unavailable, the script skips it without error)

Install:

```bash
python3 -m pip install numpy pandas scikit-learn scipy matplotlib
```

## Getting the datasets

The datasets are not included in this repo. Download them from Kaggle and place the CSVs in the same folder as the Python scripts.

**IEEE-CIS Fraud Detection** (~120 MB zipped, ~700 MB uncompressed):

1. Accept the competition rules at https://www.kaggle.com/competitions/ieee-fraud-detection/rules
2. Download from https://www.kaggle.com/competitions/ieee-fraud-detection/data
3. Unzip; you only need `train_transaction.csv`.

**Sparkov synthetic** (~280 MB):

1. Download from https://www.kaggle.com/datasets/kartik2112/fraud-detection
2. Unzip; you'll get `fraudTrain.csv` and `fraudTest.csv`. The script auto-concatenates them.

The dataset is detected automatically based on the CSV column names. Place only one dataset's CSVs in the folder at a time.

## Reproducing the paper's results

### Table 2 and Table 3 (main IEEE-CIS results)

```bash
python3 run_experiments.py --sample 0 --min-tx 5
```

Runs all methods on the full IEEE-CIS dataset (filtered to customers with ≥ 5 transactions). Takes ~20-30 minutes. Writes `results/results.csv` and `results/pr_curve.png`.

### Table 4 (alpha sweep on IEEE-CIS, 5 seeds)

```bash
python3 sweep_alpha.py --seeds 5 --sample 0 --min-tx 5
```

Runs the full α sweep across 5 random seeds. Takes ~15-20 minutes. Writes `results/alpha_sweep.csv`.

### Multi-seed validation (paired t-tests)

```bash
python3 validate_silif.py --seeds 5 --sample 0 --min-tx 5
```

Runs SilIF and top baselines across 5 seeds with paired t-tests. Takes ~2-3 minutes. Writes `results/multiseed_results.csv` and a summary.

### Sparkov results

Swap in the Sparkov CSVs (move IEEE files out, move Sparkov files in) and run the same three commands. The script auto-detects the dataset.

## Faster runs for development

For smoke-testing on a subsample:

```bash
python3 run_experiments.py --sample 100000 --min-tx 5
python3 sweep_alpha.py --seeds 3 --sample 100000 --min-tx 5
```

The 100K-row sample completes in 5-10 minutes and gives results in the same direction as the full run.

## Implementation notes

- Per-customer filtering (`--min-tx 5`) is applied before any other processing. This keeps customers whose transaction history is long enough for meaningful per-customer analysis.
- All methods are unsupervised. The `isFraud` label is used only for post-hoc evaluation (AUC-ROC, AUC-PR, Precision@k); no method sees labels during scoring.
- Numbers in the paper are mean across 5 seeds (42–46). Per-seed values are in the output CSVs.
- The silhouette in SilIF uses a centroid-distance approximation rather than the exact O(N²) silhouette, for scalability. The exact silhouette is intractable at the dataset sizes used.

## Citation

If you find this code useful, please cite the paper:

```bibtex
@article{Gopalakrishnan2026SilIF,
  title   = {SilIF: Silhouette-Augmented Isolation Forest for Unsupervised Transaction Fraud Detection},
  author  = {Gopalakrishnan, Venkatakrishnan},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

(BibTeX entry will be updated with the actual arXiv ID after the preprint is posted.)

## Contact

Questions or issues: open a GitHub issue, or email venky@uchicago.edu.

## License

MIT License. See `LICENSE`.
