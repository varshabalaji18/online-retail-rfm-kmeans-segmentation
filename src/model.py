"""Train, profile, and persist a production-shaped clustering pipeline."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def make_pipeline(n_clusters: int = 4, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)),
    ])


def train_model(rfm: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    missing = set(RFM_COLUMNS).difference(rfm.columns)
    if missing:
        raise ValueError(f"RFM matrix missing columns: {sorted(missing)}")
    if not np.isfinite(rfm[RFM_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("RFM values must be finite and non-missing")
    if (rfm[RFM_COLUMNS] < 0).any().any():
        raise ValueError("RFM values must be non-negative before log transformation")
    if not 2 <= n_clusters < len(rfm) or len(rfm[RFM_COLUMNS].drop_duplicates()) < n_clusters:
        raise ValueError("Need at least k distinct customer profiles and more than k customers")
    pipeline = make_pipeline(n_clusters, random_state)
    labels = pipeline.fit_predict(rfm[RFM_COLUMNS])
    scored = rfm.copy()
    scored["Cluster"] = labels
    profiles = scored.groupby("Cluster")[RFM_COLUMNS].agg(["count", "mean", "median"]).round(2)
    return pipeline, scored, profiles


def save_artifacts(pipeline: Pipeline, scored: pd.DataFrame, profiles: pd.DataFrame,
                   output_dir: str | Path, snapshot_date: str | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "rfm_kmeans_pipeline.joblib")
    scored.to_csv(output_dir / "customer_segments.csv", index=False)
    profiles.to_csv(output_dir / "cluster_profiles.csv")
    metadata = {"rfm_columns": RFM_COLUMNS, "snapshot_date": snapshot_date,
                "n_clusters": int(pipeline.named_steps["cluster"].n_clusters),
                "random_state": int(pipeline.named_steps["cluster"].random_state)}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
