"""Business-facing interpretation and validation helpers."""
import pandas as pd
from sklearn.metrics import silhouette_score


def name_clusters(scored: pd.DataFrame) -> dict[int, str]:
    """Assign stable, human-readable labels from relative RFM behavior."""
    profile = scored.groupby("Cluster")[['Recency', 'Frequency', 'Monetary']].mean()
    labels: dict[int, str] = {}
    for cluster, row in profile.iterrows():
        high_value = row.Monetary >= profile.Monetary.quantile(0.75) and row.Frequency >= profile.Frequency.median()
        at_risk = row.Recency >= profile.Recency.quantile(0.75) and row.Monetary >= profile.Monetary.median()
        recent_low = row.Recency <= profile.Recency.quantile(0.50) and row.Monetary <= profile.Monetary.median()
        if high_value:
            label = "High-Value Loyal"
        elif at_risk:
            label = "At-Risk Valuable"
        elif recent_low:
            label = "Recent / Low-Spend"
        else:
            label = "Core / Mid-Value"
        labels[int(cluster)] = label
    return labels


def evaluate_clustering(scored: pd.DataFrame, transformed_features) -> dict[str, float]:
    """Return a compact validation payload for the model card/dashboard."""
    if scored["Cluster"].nunique() < 2:
        return {"silhouette_score": float("nan")}
    return {"silhouette_score": float(silhouette_score(transformed_features, scored["Cluster"]))}
