"""Calcula las métricas descriptivas de K-Means para la ficha final."""
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ["watch_hours", "last_login_days", "avg_watch_time_per_day", "number_of_profiles"]

if __name__ == "__main__":
    data = pd.read_csv(ROOT / "datos" / "netflix_validado.csv")
    x = StandardScaler().fit_transform(data[VARIABLES])
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=6, n_init=10, random_state=42).fit(x)
        silhouette = silhouette_score(x, model.labels_)
    print(f"n={len(x)} K=6 inercia={model.inertia_:.2f} silueta={silhouette:.4f}")
