"""Candidate K comparison and initialization stability (not temporal validation)."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score
from .model import train_model, RFM_COLUMNS


def compare_clusters(rfm, candidates=range(2, 7)):
    rows = []
    for k in candidates:
        if k >= len(rfm) or k > len(rfm[RFM_COLUMNS].drop_duplicates()):
            continue
        model, scored, _ = train_model(rfm, k)
        features = model[:-1].transform(rfm[RFM_COLUMNS])
        stability = []
        for seed in (7, 21, 84):
            _, repeat, _ = train_model(rfm, k, seed)
            stability.append(adjusted_rand_score(scored.Cluster, repeat.Cluster))
        rows.append({"k": k, "silhouette": silhouette_score(
            features, scored.Cluster, sample_size=min(2000, len(rfm)), random_state=42),
            "inertia": model[-1].inertia_, "seed_stability_ari_mean": np.mean(stability),
            "minimum_cluster_share": scored.Cluster.value_counts(normalize=True).min()})
    if not rows:
        raise ValueError("Too few distinct customers for cluster validation")
    return pd.DataFrame(rows)
