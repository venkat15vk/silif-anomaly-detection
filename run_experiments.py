"""
Cross-Cluster Transaction Risk Assessment — Experiments
========================================================

End-to-end script that:
  1. Loads the PaySim dataset (Kaggle: ealaxi/paysim1).
  2. Implements the two-level clustering method from the patent.
  3. Implements baselines: Isolation Forest, LOF, CBLOF, Global-KMeans.
  4. Evaluates all methods with AUC-ROC, AUC-PR, Precision@k, Recall@k.
  5. Saves results (CSV + plots) into ./results/.

How to run
----------
1. Save your PaySim CSV in this same folder. The Kaggle file is usually named:
       PS_20174392719_1491204439457_log.csv
   Any *.csv in the folder containing the expected columns will be picked up.

2. Install deps (one-time):
       pip install numpy pandas scikit-learn pyod matplotlib tqdm

3. Run:
       python run_experiments.py

Optional flags (edit CONFIG below, or pass via CLI):
   --sample N       Subsample to N transactions (default: 200_000 for speed)
   --seed S         Random seed (default: 42)
   --posting-period {hour,day,week}  How to bucket time (default: day)

Notes
-----
- PaySim has ~6.3M rows; we subsample by default so the script finishes in
  ~5-15 minutes on a laptop. For final paper numbers, run with --sample 0
  (use full dataset) on a machine with enough RAM (~16GB recommended).
- All randomness is seeded; results are reproducible.
- The method follows US 2025/0094989 A1, operations 202-216.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

DEFAULT_SAMPLE = 200_000          # set to 0 for full dataset
DEFAULT_SEED = 42
DEFAULT_POSTING_PERIOD = "day"    # 'hour', 'day', or 'week'
INTRA_CLUSTERS_PER_CUSTOMER = 3   # K in per-customer K-means
CROSS_CLUSTERS = 8                # K in cross-customer K-means within a posting period
MIN_TX_PER_CUSTOMER = 3           # customers with fewer txns get a fallback score
RISK_WEIGHT_INTRA = 0.5           # weight on intra-cluster relationship score
RISK_WEIGHT_CROSS = 0.5           # weight on cross-cluster deviation
PRECISION_K_VALUES = [50, 100, 500, 1000]

# -----------------------------------------------------------------------------
# DATA LOADING (auto-detects PaySim or IEEE-CIS)
# -----------------------------------------------------------------------------

PAYSIM_COLS  = {"step", "type", "amount", "nameOrig", "isFraud"}
IEEE_COLS    = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1"}
SPARKOV_COLS = {"cc_num", "amt", "category", "merchant", "unix_time", "is_fraud"}


def detect_dataset(csv_path: Path) -> str:
    head = pd.read_csv(csv_path, nrows=2)
    cols = set(head.columns)
    if PAYSIM_COLS.issubset(cols):
        return "paysim"
    if IEEE_COLS.issubset(cols):
        return "ieee"
    if SPARKOV_COLS.issubset(cols):
        return "sparkov"
    return "unknown"


def find_csv() -> Tuple[Path, str]:
    """Find a recognized CSV in the current directory and identify the schema."""
    for c in Path(".").glob("*.csv"):
        try:
            kind = detect_dataset(c)
            if kind != "unknown":
                return c, kind
        except Exception:
            continue
    raise FileNotFoundError(
        "Could not find a PaySim or IEEE-CIS CSV in this folder.\n"
        " - PaySim:   https://www.kaggle.com/datasets/ealaxi/paysim1\n"
        " - IEEE-CIS: https://www.kaggle.com/competitions/ieee-fraud-detection/data\n"
        "   (use train_transaction.csv)"
    )


def load_paysim(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["customer"] = df["nameOrig"]
    df["amount_f"] = df["amount"].astype(float)
    df["type_id"]  = df["type"].astype("category").cat.codes
    df["bal_a"]    = df["oldbalanceOrg"].clip(lower=0)
    df["bal_b"]    = df["newbalanceOrig"].clip(lower=0)
    df["bal_c"]    = df["oldbalanceDest"].clip(lower=0)
    df["bal_d"]    = df["newbalanceDest"].clip(lower=0)
    df["posting_period_hour"] = df["step"]
    df["posting_period_day"]  = df["step"] // 24
    df["posting_period_week"] = df["step"] // (24 * 7)
    return df


def load_ieee(csv_path: Path) -> pd.DataFrame:
    """
    IEEE-CIS schema. card1 is the persistent card-level identifier — this is
    the 'customer' for our method. TransactionDT is seconds from a reference.
    """
    head = pd.read_csv(csv_path, nrows=2)
    use_cols = [c for c in
                ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
                 "ProductCD", "card1", "card2", "card3", "card5",
                 "addr1", "addr2", "C1", "C2", "C13", "C14",
                 "D1", "D2", "D15"]
                if c in head.columns]
    df = pd.read_csv(csv_path, usecols=use_cols)
    df["customer"] = df["card1"].astype(str)
    df["amount_f"] = df["TransactionAmt"].astype(float)
    df["type_id"]  = df["ProductCD"].astype("category").cat.codes if "ProductCD" in df else 0
    # Numeric features the method will use (filled, log-scaled later)
    for c in ["card2", "card3", "card5", "addr1", "addr2", "C1", "C2",
              "C13", "C14", "D1", "D2", "D15"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["bal_a"] = df.get("C1",  0.0)
    df["bal_b"] = df.get("C2",  0.0)
    df["bal_c"] = df.get("C13", 0.0)
    df["bal_d"] = df.get("C14", 0.0)
    df["posting_period_hour"] = (df["TransactionDT"] // 3600).astype(int)
    df["posting_period_day"]  = (df["TransactionDT"] // 86400).astype(int)
    df["posting_period_week"] = (df["TransactionDT"] // (86400 * 7)).astype(int)
    return df


def load_sparkov(csv_path: Path) -> pd.DataFrame:
    """
    Sparkov schema (Kartik Shenoy synthetic credit card).
    Auto-concatenates fraudTrain.csv + fraudTest.csv if both are present in
    the same folder.
      customer = cc_num (credit card number, persistent customer ID)
      time     = unix_time (seconds since epoch)
      amount   = amt
      type     = category (grocery_pos, gas_transport, ...)
    """
    folder = csv_path.parent
    train = folder / "fraudTrain.csv"
    test  = folder / "fraudTest.csv"
    parts = []
    if train.exists():
        parts.append(pd.read_csv(train))
    if test.exists():
        parts.append(pd.read_csv(test))
    if not parts:
        parts = [pd.read_csv(csv_path)]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    df["customer"] = df["cc_num"].astype(str)
    df["amount_f"] = df["amt"].astype(float)
    df["type_id"]  = df["category"].astype("category").cat.codes
    # Rename to the unified label column the rest of the script expects
    df["isFraud"]  = df["is_fraud"].astype(int)
    # Numeric features beyond amount + category: geo distance proxies if present
    for c in ["lat", "long", "merch_lat", "merch_long", "city_pop", "zip"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Map four numeric features into bal_a..bal_d (used by per_transaction_features)
    df["bal_a"] = df.get("lat",       0.0)
    df["bal_b"] = df.get("long",      0.0)
    df["bal_c"] = df.get("merch_lat", 0.0)
    df["bal_d"] = df.get("merch_long",0.0)
    # Posting periods
    df["posting_period_hour"] = (df["unix_time"] // 3600).astype(int)
    df["posting_period_day"]  = (df["unix_time"] // 86400).astype(int)
    df["posting_period_week"] = (df["unix_time"] // (86400 * 7)).astype(int)
    return df


def load_data(sample: int, seed: int, min_tx: int = 1) -> Tuple[pd.DataFrame, str]:
    csv_path, kind = find_csv()
    print(f"[data] Loading {csv_path} (schema: {kind}) ...")
    t0 = time.time()
    if kind == "paysim":
        df = load_paysim(csv_path)
    elif kind == "ieee":
        df = load_ieee(csv_path)
    elif kind == "sparkov":
        df = load_sparkov(csv_path)
    else:
        raise RuntimeError(f"Unknown dataset kind: {kind}")
    print(f"[data] Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Filter to customers with enough transactions (before subsampling)
    if min_tx > 1:
        cust_counts = df["customer"].value_counts()
        keep = cust_counts[cust_counts >= min_tx].index
        before = len(df)
        df = df[df["customer"].isin(keep)].reset_index(drop=True)
        print(f"[data] Filtered to customers with >={min_tx} txns: "
              f"{before:,} -> {len(df):,} rows, {len(keep):,} customers")

    if len(df) == 0:
        raise RuntimeError(
            f"No transactions remain after --min-tx {min_tx}. "
            f"Lower the threshold or use a different dataset."
        )

    if sample and sample < len(df):
        fraud = df[df["isFraud"] == 1]
        nonfraud = df[df["isFraud"] == 0]
        if min_tx > 1:
            # Sample whole customers so multi-tx customers keep all their txns
            nonfraud_custs = nonfraud["customer"].unique()
            np.random.seed(seed)
            target_n = max(sample - len(fraud), 1)
            avg_tx = len(nonfraud) / max(len(nonfraud_custs), 1)
            n_custs = min(len(nonfraud_custs), max(1, int(target_n / max(avg_tx, 1))))
            picked = np.random.choice(nonfraud_custs, size=n_custs, replace=False)
            nonfraud_s = nonfraud[nonfraud["customer"].isin(picked)]
        else:
            target_nonfraud = max(sample - len(fraud), 1)
            target_nonfraud = min(target_nonfraud, len(nonfraud))
            nonfraud_s = nonfraud.sample(n=target_nonfraud, random_state=seed)
        df = pd.concat([fraud, nonfraud_s]).sample(frac=1, random_state=seed).reset_index(drop=True)
        n_custs_final = df["customer"].nunique()
        print(f"[data] Subsampled to {len(df):,} rows "
              f"({len(fraud)} fraud, {len(nonfraud_s):,} non-fraud, "
              f"{n_custs_final:,} unique customers, "
              f"{len(df)/max(n_custs_final,1):.2f} txns/customer avg)")

    return df, kind


# -----------------------------------------------------------------------------
# FEATURE ENGINEERING (dataset-agnostic — uses unified column names)
# -----------------------------------------------------------------------------

def per_transaction_features(df: pd.DataFrame) -> np.ndarray:
    """Numeric features per transaction (used inside per-customer clustering).

    Uses sign-preserving log scaling for features that can be negative
    (e.g., geo coordinates in Sparkov) and clips amount to non-negative."""
    def signed_log(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sign(x) * np.log1p(np.abs(x))

    feats = pd.DataFrame({
        "amount": np.log1p(np.nan_to_num(df["amount_f"].clip(lower=0).values, nan=0.0)),
        "type":   np.nan_to_num(df["type_id"].values.astype(float), nan=0.0),
        "f1":     signed_log(df["bal_a"].values),
        "f2":     signed_log(df["bal_b"].values),
        "f3":     signed_log(df["bal_c"].values),
        "f4":     signed_log(df["bal_d"].values),
    })
    return feats.values


def per_customer_features(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """Per-(customer, posting-period) aggregate features for the cross-cluster step."""
    grp = df.groupby(["customer", period_col])
    agg = grp.agg(
        tx_count=("amount_f", "size"),
        amt_sum=("amount_f", "sum"),
        amt_mean=("amount_f", "mean"),
        amt_std=("amount_f", "std"),
        type_mode=("type_id", lambda x: x.mode().iloc[0] if len(x) else 0),
    ).fillna(0.0).reset_index()
    agg["amt_sum"]  = np.log1p(agg["amt_sum"])
    agg["amt_mean"] = np.log1p(agg["amt_mean"])
    agg["amt_std"]  = np.log1p(agg["amt_std"])
    agg["tx_count_log"] = np.log1p(agg["tx_count"])
    return agg


# -----------------------------------------------------------------------------
# METHOD: TWO-LEVEL CROSS-CLUSTER RISK SCORING (the patent's method)
# -----------------------------------------------------------------------------

@dataclass
class MethodConfig:
    intra_k: int = INTRA_CLUSTERS_PER_CUSTOMER
    cross_k: int = CROSS_CLUSTERS
    k_neighbors: int = 3                              # k for intra/inter k-NN
    period_col: str = "posting_period_day"
    weight_intra: float = RISK_WEIGHT_INTRA
    weight_cross: float = RISK_WEIGHT_CROSS
    min_tx_per_customer: int = MIN_TX_PER_CUSTOMER
    seed: int = DEFAULT_SEED


def compute_intra_relationship_scores(df: pd.DataFrame, X: np.ndarray,
                                      cfg: MethodConfig) -> np.ndarray:
    """
    Per-customer relationship scoring using SILHOUETTE + k-NN (the patent's
    core innovation):

    For each customer:
      1. Cluster the customer's transactions (K-means with intra_k clusters).
      2. For each transaction i in cluster C:
           a(i) = mean distance to k nearest neighbors IN cluster C   (intra-NN)
           b(i) = mean distance to k nearest neighbors in the NEAREST OTHER
                  cluster (inter-NN)
           silhouette s(i) = (b(i) - a(i)) / max(a(i), b(i))
      3. Anomaly score = 1 - s(i)   (range [0, 2]; higher = more anomalous;
         a point that fits poorly in its own cluster but matches another gets
         silhouette near -1, anomaly score near 2)

    Silhouette REPLACES distance-to-centroid as the relationship score.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scores = np.zeros(len(df))
    customer_groups = df.groupby("customer").indices

    print(f"[method] Computing per-customer SILHOUETTE+kNN relationship scores "
          f"for {len(customer_groups):,} customers ...")

    k_nn = cfg.k_neighbors

    for ci, (cust, idxs) in enumerate(customer_groups.items()):
        if ci % 50_000 == 0 and ci > 0:
            print(f"  ... {ci:,} customers processed")

        n = len(idxs)
        Xc = X[idxs]

        # Single-transaction customer: no neighbors, no clusters; neutral score
        if n < 2:
            scores[idxs] = 0.0
            continue

        scaler = StandardScaler()
        # StandardScaler with a single row would produce zeros; protect it
        Xc_s = scaler.fit_transform(Xc) if n > 1 else Xc

        # Need at least 2 clusters for silhouette to mean anything.
        # Cap k by n so we never request more clusters than points.
        k = min(cfg.intra_k, n)
        if k < 2:
            # Only one cluster possible: silhouette is undefined.
            # Fall back to relative distance from customer's own mean.
            mu = Xc_s.mean(axis=0)
            d = np.linalg.norm(Xc_s - mu, axis=1)
            scores[idxs] = d / (d.std() + 1e-9)
            continue

        km = KMeans(n_clusters=k, n_init=3, random_state=cfg.seed)
        labels = km.fit_predict(Xc_s)

        # If KMeans collapsed (one non-empty cluster), fall back
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            mu = Xc_s.mean(axis=0)
            d = np.linalg.norm(Xc_s - mu, axis=1)
            scores[idxs] = d / (d.std() + 1e-9)
            continue

        # Compute silhouette per point using k-NN within and across clusters.
        # For small per-customer n this is cheap; we do it directly.
        sil = np.zeros(n)
        # Precompute pairwise distances within this customer's transactions
        from scipy.spatial.distance import cdist
        D = cdist(Xc_s, Xc_s)  # n x n

        for i in range(n):
            own = labels[i]
            # a(i): mean distance to k_nn nearest neighbors in own cluster
            own_mask = (labels == own)
            own_mask[i] = False  # exclude self
            own_dists = D[i, own_mask]
            if len(own_dists) == 0:
                # Singleton cluster: treat a(i) as 0 (point IS its own cluster)
                a_i = 0.0
            else:
                kk = min(k_nn, len(own_dists))
                a_i = np.partition(own_dists, kk - 1)[:kk].mean()

            # b(i): for each OTHER cluster, mean distance to k_nn nearest in it;
            # take the MIN across other clusters (the "nearest other cluster")
            b_candidates = []
            for c in unique_labels:
                if c == own:
                    continue
                other_mask = (labels == c)
                other_dists = D[i, other_mask]
                if len(other_dists) == 0:
                    continue
                kk = min(k_nn, len(other_dists))
                b_candidates.append(np.partition(other_dists, kk - 1)[:kk].mean())
            if not b_candidates:
                b_i = a_i  # no other cluster: silhouette = 0
            else:
                b_i = min(b_candidates)

            denom = max(a_i, b_i)
            sil[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0

        # Convert to anomaly score: low/negative silhouette => high anomaly
        # 1 - sil maps [-1, 1] -> [2, 0]; higher = more anomalous
        scores[idxs] = 1.0 - sil

    # Standardize so the weighted combination later is balanced
    scores = (scores - scores.mean()) / (scores.std() + 1e-9)
    return scores


def compute_cross_cluster_scores(df: pd.DataFrame, cfg: MethodConfig) -> np.ndarray:
    """
    Operations 210-212 of the patent:
    Within each posting period, cluster CUSTOMERS by their aggregate attributes
    (transaction volume, frequency, type mix). Compute each customer's distance
    to their cluster's centroid AND to all other cluster centroids; combine into
    a cross-cluster deviation score, then broadcast back to each transaction.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print(f"[method] Computing cross-customer cluster deviations "
          f"per posting period ({cfg.period_col}) ...")

    cust_feats = per_customer_features(df, cfg.period_col)
    feat_cols = ["tx_count_log", "amt_sum", "amt_mean", "amt_std", "type_mode"]

    customer_scores: Dict[Tuple[str, int], float] = {}

    for period, sub in cust_feats.groupby(cfg.period_col):
        if len(sub) < cfg.cross_k * 2:
            # Too few customers in this period — give neutral score
            for _, row in sub.iterrows():
                customer_scores[(row["customer"], period)] = 0.0
            continue
        X = StandardScaler().fit_transform(sub[feat_cols].values)
        k = min(cfg.cross_k, len(sub))
        km = KMeans(n_clusters=k, n_init=3, random_state=cfg.seed)
        labels = km.fit_predict(X)

        # For each customer: distance to own centroid + mean distance to other centroids
        for i, (_, row) in enumerate(sub.iterrows()):
            own_label = labels[i]
            own_d = np.linalg.norm(X[i] - km.cluster_centers_[own_label])
            other_d = [np.linalg.norm(X[i] - km.cluster_centers_[c])
                       for c in range(k) if c != own_label]
            # Patent: risk score evaluates relationship vs. own centroid AND others' centroids
            # If a customer is FAR from own centroid but CLOSE to another, that's anomalous.
            # We use: own_d - min(other_d) ; high values = customer is an outlier in own group
            mean_other = np.mean(other_d) if other_d else 0.0
            customer_scores[(row["customer"], period)] = own_d - 0.5 * mean_other

    # Broadcast customer-period scores back to each transaction
    df_key = list(zip(df["customer"], df[cfg.period_col]))
    out = np.array([customer_scores.get(k, 0.0) for k in df_key])
    out = (out - out.mean()) / (out.std() + 1e-9)
    return out


def cross_cluster_risk_method(df: pd.DataFrame, cfg: MethodConfig) -> np.ndarray:
    """Combined risk score per transaction (patent operation 212)."""
    X = per_transaction_features(df)
    intra = compute_intra_relationship_scores(df, X, cfg)
    cross = compute_cross_cluster_scores(df, cfg)
    # Patent claim 12: weighted combination of intra and cross
    raw = cfg.weight_intra * intra + cfg.weight_cross * cross
    # Logarithmic accentuation (patent paragraph [0064])
    sign = np.sign(raw)
    risk = sign * np.log1p(np.abs(raw))
    return risk


def silk_intra_only(df: pd.DataFrame, cfg: MethodConfig) -> np.ndarray:
    """
    SilK (intra-only): per-customer silhouette+kNN, no cross-cluster step.
    Tests whether the cross-cluster weighting was the problem in the
    two-level version.
    """
    X = per_transaction_features(df)
    return compute_intra_relationship_scores(df, X, cfg)


def silif_method(df: pd.DataFrame, seed: int, n_estimators: int = 100,
                 n_structure_clusters: int = 8, alpha: float = 0.5) -> np.ndarray:
    """
    Silhouette-Augmented Isolation Forest (SilIF).

    Idea: Isolation Forest already scores anomalies by average path length
    across many trees. We add a silhouette layer:

      1. Train IF as usual.
      2. For each point, build a "structural fingerprint": its path length
         in every tree (vector of length n_estimators).
      3. Cluster points by their fingerprints (K-means). Points landing in
         similar isolation paths form a "structural cluster" — this is what
         the silhouette sees.
      4. Compute classical silhouette s(i) = (b(i) - a(i)) / max(a(i), b(i))
         where a(i) = mean distance to OWN structural cluster's points
               b(i) = mean distance to nearest OTHER structural cluster's points
         Distances are taken in the fingerprint space.
      5. Final anomaly = (base IF score) + alpha * (1 - silhouette)

    Higher = more anomalous. Both the base score and (1 - silhouette) are
    z-standardized before combining so alpha has a meaningful scale.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances_argmin_min

    print(f"[method] Silhouette-Augmented Isolation Forest (SilIF) ...")
    X = per_transaction_features(df)
    n = len(X)

    # Step 1: IF
    print(f"  [silif] training IF with {n_estimators} trees on {n:,} points ...")
    iso = IsolationForest(n_estimators=n_estimators, contamination="auto",
                          random_state=seed, n_jobs=-1)
    iso.fit(X)
    base_score = -iso.score_samples(X)  # higher = more anomalous

    # Step 2: per-tree path-length fingerprint
    print(f"  [silif] computing per-tree fingerprints ...")
    # For each tree, get the depth at which each point was isolated.
    # IsolationForest exposes per-estimator decision_path / decision_function,
    # but the cleanest cross-version way is the per-tree leaves_depth.
    fingerprint = np.zeros((n, n_estimators), dtype=np.float32)
    for t_idx, est in enumerate(iso.estimators_):
        leaves = est.apply(X)
        # Depth of each sample = node_count_to_leaf from tree_
        # sklearn exposes tree_.compute_node_depths() in newer versions;
        # universally available: walk via decision_path is O(n*nodes), too slow.
        # Use the node 'n_node_samples' approach: depth ~ log2(n_node_samples_at_root / n_node_samples_at_leaf).
        # Simpler: use the estimator's decision_function as a scalar fingerprint
        # if path-length isn't directly accessible.
        # Fastest and stable: get depth from tree_ structure.
        tree = est.tree_
        depths = _node_depths(tree)
        fingerprint[:, t_idx] = depths[leaves]

    # Step 3: cluster fingerprints
    print(f"  [silif] clustering {n_structure_clusters} structural groups ...")
    fp_scaled = StandardScaler().fit_transform(fingerprint)
    # MiniBatchKMeans for speed on large n
    if n > 50_000:
        km = MiniBatchKMeans(n_clusters=n_structure_clusters,
                             random_state=seed, batch_size=4096, n_init=3)
    else:
        km = KMeans(n_clusters=n_structure_clusters, n_init=3, random_state=seed)
    cluster_labels = km.fit_predict(fp_scaled)

    # Step 4: silhouette — using cluster centroids as proxies for speed
    # a(i) = dist to own centroid, b(i) = dist to nearest other centroid
    # (Approximation of classical silhouette that scales to large n.)
    print(f"  [silif] computing silhouette scores ...")
    centroids = km.cluster_centers_
    sil = np.zeros(n)
    for i in range(n):
        own = cluster_labels[i]
        own_d = np.linalg.norm(fp_scaled[i] - centroids[own])
        other_ds = [np.linalg.norm(fp_scaled[i] - centroids[c])
                    for c in range(n_structure_clusters) if c != own]
        b_i = min(other_ds) if other_ds else own_d
        denom = max(own_d, b_i)
        sil[i] = (b_i - own_d) / denom if denom > 1e-12 else 0.0

    sil_anomaly = 1.0 - sil  # high = anomalous

    # Step 5: combine (z-standardize both, then weighted sum)
    base_z = (base_score - base_score.mean()) / (base_score.std() + 1e-9)
    sil_z = (sil_anomaly - sil_anomaly.mean()) / (sil_anomaly.std() + 1e-9)
    return base_z + alpha * sil_z


def _node_depths(tree) -> np.ndarray:
    """Compute depth of each node in a sklearn decision tree."""
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    depth = np.zeros(n_nodes, dtype=np.int32)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        if children_left[node] != -1:
            stack.append((children_left[node], d + 1))
        if children_right[node] != -1:
            stack.append((children_right[node], d + 1))
    return depth


# -----------------------------------------------------------------------------
# BASELINES
# -----------------------------------------------------------------------------

def baseline_isolation_forest(df: pd.DataFrame, seed: int) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    print("[baseline] Isolation Forest ...")
    X = per_transaction_features(df)
    clf = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=seed, n_jobs=-1)
    clf.fit(X)
    return -clf.score_samples(X)  # higher = more anomalous


def baseline_lof(df: pd.DataFrame, seed: int) -> np.ndarray:
    from sklearn.neighbors import LocalOutlierFactor
    print("[baseline] LOF ...")
    X = per_transaction_features(df)
    lof = LocalOutlierFactor(n_neighbors=20, n_jobs=-1)
    lof.fit_predict(X)
    return -lof.negative_outlier_factor_


def baseline_knn_distance(df: pd.DataFrame, seed: int, k: int = 5) -> np.ndarray:
    """
    Unsupervised: anomaly score = mean distance to k nearest neighbors in the
    full transaction space. Classical and strong unsupervised baseline.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    print(f"[baseline] kNN-distance (k={k}) ...")
    X = StandardScaler().fit_transform(per_transaction_features(df))
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    nn.fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, 1:].mean(axis=1)  # exclude self


def baseline_hbos(df: pd.DataFrame, seed: int, n_bins: int = 20) -> np.ndarray:
    """
    Histogram-Based Outlier Score (Goldstein & Dengel, 2012).
    Unsupervised, very fast, strong baseline for high-dim data.
    Pure-numpy implementation; uses pyod if available.
    """
    print("[baseline] HBOS ...")
    try:
        from pyod.models.hbos import HBOS
        X = per_transaction_features(df)
        clf = HBOS(n_bins=n_bins)
        clf.fit(X)
        return clf.decision_scores_
    except Exception:
        pass
    # Fallback: per-feature histogram log-density
    X = per_transaction_features(df)
    n, d = X.shape
    scores = np.zeros(n)
    for j in range(d):
        col = X[:, j]
        hist, edges = np.histogram(col, bins=n_bins)
        # Density per bin, normalized to max=1
        density = hist / max(hist.max(), 1)
        # Assign each point to its bin
        bin_idx = np.clip(np.digitize(col, edges[:-1]) - 1, 0, n_bins - 1)
        # Per-feature score = log(1/density). High density -> low score.
        scores += np.log(1.0 / np.maximum(density[bin_idx], 1e-6))
    return scores


def baseline_ecod(df: pd.DataFrame, seed: int) -> np.ndarray:
    """
    ECOD (Li et al., 2022): Empirical Cumulative Distribution-based Outlier
    Detection. Unsupervised, parameter-free, strong recent baseline.
    Uses pyod if available; falls back to a pure-numpy implementation.
    """
    print("[baseline] ECOD ...")
    try:
        from pyod.models.ecod import ECOD
        X = per_transaction_features(df)
        clf = ECOD()
        clf.fit(X)
        return clf.decision_scores_
    except Exception:
        pass
    # Fallback: per-feature empirical CDF tail probability, summed in log-space
    X = per_transaction_features(df)
    n, d = X.shape
    scores = np.zeros(n)
    for j in range(d):
        col = X[:, j]
        ranks = np.argsort(np.argsort(col)) + 1   # 1..n
        # Two-sided tail: distance from median rank, expressed as small prob
        ecdf = ranks / (n + 1)
        tail = np.minimum(ecdf, 1 - ecdf)         # closer to 0 = more extreme
        scores += -np.log(np.maximum(tail, 1e-9))
    return scores


def baseline_cblof(df: pd.DataFrame, seed: int) -> np.ndarray:
    """Cluster-Based Local Outlier Factor (He et al., 2003)."""
    print("[baseline] CBLOF ...")
    try:
        from pyod.models.cblof import CBLOF
        X = per_transaction_features(df)
        clf = CBLOF(n_clusters=8, contamination=0.01, random_state=seed,
                    check_estimator=False)
        clf.fit(X)
        return clf.decision_scores_
    except Exception as e:
        print(f"  [warn] CBLOF unavailable ({e}); skipping.")
        return None


def baseline_global_kmeans(df: pd.DataFrame, seed: int) -> np.ndarray:
    """
    Sanity baseline: cluster ALL transactions globally with K-means,
    score each by distance to its assigned centroid. Essentially the
    single-level version of our method (no per-customer, no cross-cluster).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    print("[baseline] Global K-means distance ...")
    X = StandardScaler().fit_transform(per_transaction_features(df))
    km = KMeans(n_clusters=8, n_init=5, random_state=seed)
    labels = km.fit_predict(X)
    d = np.array([np.linalg.norm(X[i] - km.cluster_centers_[labels[i]])
                  for i in range(len(X))])
    return d


# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------

def evaluate(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    out = {
        "AUC_ROC": roc_auc_score(y_true, scores),
        "AUC_PR":  average_precision_score(y_true, scores),
    }
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n_pos = y_true.sum()
    for k in PRECISION_K_VALUES:
        k_eff = min(k, len(y_sorted))
        out[f"P@{k}"] = y_sorted[:k_eff].sum() / k_eff
        out[f"R@{k}"] = y_sorted[:k_eff].sum() / max(n_pos, 1)
    return out


# -----------------------------------------------------------------------------
# DRIVER
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="Subsample size (0 = full dataset)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--posting-period", choices=["hour", "day", "week"],
                        default=DEFAULT_POSTING_PERIOD)
    parser.add_argument("--min-tx", type=int, default=1,
                        help="Restrict to customers with >= this many transactions")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # ---- Load data
    df, dataset_kind = load_data(sample=args.sample, seed=args.seed, min_tx=args.min_tx)
    y = df["isFraud"].values.astype(int)
    print(f"[data] {len(df):,} txns, {y.sum():,} fraud "
          f"({100*y.mean():.3f}% positive rate)")

    cfg = MethodConfig(
        period_col=f"posting_period_{args.posting_period}",
        seed=args.seed,
    )

    # ---- Run all methods
    results: Dict[str, Dict[str, float]] = {}

    # --- OUR METHODS ---
    t0 = time.time()
    risk_silk2 = cross_cluster_risk_method(df, cfg)
    print(f"[time] SilK (two-level, patent): {time.time()-t0:.1f}s")
    results["SilK-2level (patent)"] = evaluate(risk_silk2, y)

    t0 = time.time()
    risk_silk1 = silk_intra_only(df, cfg)
    print(f"[time] SilK (intra-only): {time.time()-t0:.1f}s")
    results["SilK-intra"] = evaluate(risk_silk1, y)

    t0 = time.time()
    risk_silif = silif_method(df, args.seed)
    print(f"[time] SilIF: {time.time()-t0:.1f}s")
    results["SilIF (silhouette-augmented IF)"] = evaluate(risk_silif, y)

    t0 = time.time()
    results["IsolationForest"] = evaluate(baseline_isolation_forest(df, args.seed), y)
    print(f"[time] IF: {time.time()-t0:.1f}s")

    t0 = time.time()
    results["HBOS"] = evaluate(baseline_hbos(df, args.seed), y)
    print(f"[time] HBOS: {time.time()-t0:.1f}s")

    t0 = time.time()
    results["ECOD"] = evaluate(baseline_ecod(df, args.seed), y)
    print(f"[time] ECOD: {time.time()-t0:.1f}s")

    t0 = time.time()
    results["GlobalKMeans"] = evaluate(baseline_global_kmeans(df, args.seed), y)
    print(f"[time] Global K-means: {time.time()-t0:.1f}s")

    # CBLOF only if pyod is available
    cblof_scores = baseline_cblof(df, args.seed)
    if cblof_scores is not None:
        results["CBLOF"] = evaluate(cblof_scores, y)

    # kNN-distance and LOF are O(N^2)-ish; only run if dataset is small enough
    if len(df) <= 100_000:
        t0 = time.time()
        results["kNN-distance"] = evaluate(baseline_knn_distance(df, args.seed), y)
        print(f"[time] kNN-distance: {time.time()-t0:.1f}s")

        t0 = time.time()
        results["LOF"] = evaluate(baseline_lof(df, args.seed), y)
        print(f"[time] LOF: {time.time()-t0:.1f}s")
    else:
        print("[baseline] Skipping kNN and LOF (dataset too large; "
              "rerun with --sample 50000 for those)")

    # ---- Report
    res_df = pd.DataFrame(results).T
    res_df = res_df.sort_values("AUC_PR", ascending=False)
    print("\n=== RESULTS ===")
    print(res_df.round(4).to_string())

    res_df.to_csv(out_dir / "results.csv")
    print(f"\n[out] Wrote results to {out_dir / 'results.csv'}")

    # ---- Plot
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        plt.figure(figsize=(9, 6))
        for name, scores in [
            ("SilIF (ours)",            risk_silif),
            ("SilK-intra (ours)",       risk_silk1),
            ("SilK-2level (patent)",    risk_silk2),
            ("IsolationForest",         baseline_isolation_forest(df, args.seed)),
            ("GlobalKMeans",            baseline_global_kmeans(df, args.seed)),
        ]:
            p, r, _ = precision_recall_curve(y, scores)
            plt.plot(r, p, label=name, linewidth=2)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall: Fraud Detection ({dataset_kind})")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "pr_curve.png", dpi=150)
        print(f"[out] Wrote PR curve to {out_dir / 'pr_curve.png'}")
    except Exception as e:
        print(f"[plot] Skipped plot ({e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
