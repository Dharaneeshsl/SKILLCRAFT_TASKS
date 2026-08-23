"""K-Means customer segmentation for the Mall Customers dataset.

Run from any working directory:
    python "TASK 2/kmeans_clustering.py"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]


def run(data_dir: Path, max_k: int = 10, selected_k: int = 5, random_state: int = 42) -> dict[str, object]:
    data_path = data_dir / "Mall_Customers.csv"
    df = pd.read_csv(data_path)
    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if not 2 <= selected_k <= max_k:
        raise ValueError("selected_k must be between 2 and max_k")

    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ks = list(range(1, max_k + 1))
    inertias: list[float] = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        model.fit(X_scaled)
        inertias.append(float(model.inertia_))

    plt.figure(figsize=(8, 4.5))
    plt.plot(ks, inertias, marker="o")
    plt.axvline(selected_k, color="tab:red", linestyle="--", label=f"Selected k={selected_k}")
    plt.title("Elbow Method for Customer Segmentation")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-cluster sum of squares")
    plt.xticks(ks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "elbow_method.png", dpi=160)
    plt.close()

    model = KMeans(n_clusters=selected_k, n_init=20, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    silhouette = float(silhouette_score(X_scaled, labels))

    output = df.copy()
    output["Cluster"] = labels + 1
    output.to_csv(data_dir / "customer_segments.csv", index=False)

    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap("tab10")
    for cluster_id in range(selected_k):
        points = X[labels == cluster_id]
        plt.scatter(
            points[FEATURES[0]],
            points[FEATURES[1]],
            s=55,
            color=colors(cluster_id),
            alpha=0.8,
            label=f"Cluster {cluster_id + 1}",
        )
    plt.scatter(
        centers_original[:, 0],
        centers_original[:, 1],
        s=220,
        color="black",
        marker="X",
        label="Centroids",
    )
    plt.title("Customer Segments")
    plt.xlabel(FEATURES[0])
    plt.ylabel(FEATURES[1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "kmeans_clusters.png", dpi=160)
    plt.close()

    metrics = {
        "rows": int(len(df)),
        "selected_k": selected_k,
        "silhouette_score": silhouette,
        "inertia": float(model.inertia_),
        "cluster_sizes": {str(i + 1): int((labels == i).sum()) for i in range(selected_k)},
        "centroids": [
            {FEATURES[0]: float(center[0]), FEATURES[1]: float(center[1])}
            for center in centers_original
        ],
    }
    with (data_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved customer assignments to {data_dir / 'customer_segments.csv'}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--selected-k", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, max_k=args.max_k, selected_k=args.selected_k)
