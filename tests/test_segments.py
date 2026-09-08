import pandas as pd
from src.segments import name_clusters

def test_cluster_names_cover_every_cluster():
    scored = pd.DataFrame({"Cluster": [0, 0, 1, 1], "Recency": [5, 10, 100, 120], "Frequency": [10, 8, 1, 2], "Monetary": [1000, 900, 100, 120]})
    labels = name_clusters(scored)
    assert set(labels) == {0, 1}
    assert all(isinstance(value, str) and value for value in labels.values())
