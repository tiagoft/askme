from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from askme.rtp.make_collection_index import make_faiss_index
from askme.utils import (
    kmeans_with_faiss,
    select_n_random_indices,
    vote_k_sampling,
    fast_votek,
)


def generate_dataset() -> np.ndarray:
    centers = [
        [0, 0],
        [-1, 1],
        [1, -1],
    ]
    cluster_std = [0.1, 0.5, 0.9]
    n_per_cluster = [2000, 2000, 2000]
    s = np.vstack([
        np.random.normal(loc=centers[i],
                         scale=cluster_std[i],
                         size=(n_per_cluster[i], 2))
        for i in range(len(centers))
    ])
    return s


def main():
    X = generate_dataset()

    print("Making faiss index")
    faiss_index = faiss.IndexFlatL2(2)
    faiss_index.add(X.astype('float32'))

    n_clusters = 10

    print("Random selection")
    random_indices = select_n_random_indices(total_size=X.shape[0],
                                             n_select=n_clusters,
                                             seed=42)
    print("Vote-k sampling")
    votek_indices = vote_k_sampling(
        faiss_index,
        X,
        n_clusters=n_clusters,
        k_neighbors=30,
    )
    fast_votek_indices = fast_votek(
        X,
        select_num=n_clusters,
        k=30,
    )
    print("Fast Vote-k sampling indices:", fast_votek_indices)
    print("Vote-k sampling indices (ours):", votek_indices)
    print("KMeans with Faiss")
    kmeans_indices = kmeans_with_faiss(
        faiss_index,
        X,
        n_clusters=n_clusters,
        use_gpu=False,
        seed=42,
        niter=10,
        spherical=False,
    )

    print("Plotting results")
    # quick plot
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c='b', s=30, alpha=0.6, label='Data Points')
    plt.scatter(X[random_indices, 0],
                X[random_indices, 1],
                c='r',
                s=200,
                marker='X',
                label='Random Selection')
    plt.scatter(X[votek_indices, 0],
                X[votek_indices, 1],
                c='g',
                s=200,
                marker='D',
                label='Vote-k Sampling')
    plt.scatter(X[fast_votek_indices, 0],
                X[fast_votek_indices, 1],
                c='purple',
                s=200,
                marker='o',
                label='Fast Vote-k Sampling')
    plt.scatter(X[kmeans_indices, 0],
                X[kmeans_indices, 1],
                c='orange',
                s=200,
                marker='P',
                label='KMeans with Faiss')
    plt.legend()
    plt.title("2D blobs with significant intersection")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "demo_selection.png")


if __name__ == "__main__":
    main()
