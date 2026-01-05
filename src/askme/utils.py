from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def chunk_text(
    text: str,
    chunk_size: int = 350,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into chunks of specified word count with overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: Number of words per chunk (default: 350)
        overlap: Number of words to overlap between chunks (default: 50)
    
    Returns:
        List of text chunks
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) words
        start += (chunk_size - overlap)

        # Break if we've covered all words
        if end >= len(words):
            break

    return chunks


class TextEmbeddingWithChunker:

    def __init__(self,
                 model_name: str,
                 chunk_size: int = 350,
                 overlap: int = 50,
                 pooling_fn: callable = np.mean,
                 device: str = 'cpu'):

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.pooling_fn = pooling_fn

    def __call__(self, text: str) -> np.ndarray:
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        chunk_embeddings = self.model.encode_document(chunks)
        embedding = self.pooling_fn(chunk_embeddings, axis=0)
        return embedding


def kmeans_with_faiss(
    faiss_index: faiss.Index,
    X: np.ndarray,
    n_clusters: int,
) -> np.ndarray:

    kmeans = faiss.Kmeans(
        d=faiss_index.d,
        k=n_clusters,
        niter=500,
        nredo=10,
        verbose=False,
        seed=1234,
        spherical=True,
        gpu=False,
    )
    kmeans.train(X)#, init_centroids=X[:n_clusters,:])
    centroids = kmeans.centroids
    D, I = faiss_index.search(centroids, 1)  # D: distances, I: indices (shape k, 1)

    medoid_indices = I.flatten()       # Indices of medoids in your original data
    # # medoid indexes
    # # Find medoids
    # distances_to_centroids, _ = kmeans.index.search(X, k=n_clusters)
    # #print(distances_to_centroids.shape, distances_to_centroids)
    # medoid_indices = np.argmin(distances_to_centroids, axis=0)
    return medoid_indices


def true_k_medoids_faiss(
    embeddings: np.ndarray,  # normalized if needed
    n_clusters: int,
    nredo: int = 10,
    seed: int = 1234
) -> np.ndarray:
    """
    Returns indices of K unique medoids using a robust k-medoids approximation.
    """
    d = embeddings.shape[1]
    best_medoids = None
    best_inertia = np.inf

    for _ in range(nredo):
        # Randomly sample initial medoids
        np.random.seed(seed + _)
        init_indices = np.random.choice(len(embeddings), n_clusters, replace=False)
        init_centroids = embeddings[init_indices]

        # Train k-means with these as initial centroids
        kmeans = faiss.Kmeans(
            d=d,
            k=n_clusters,
            niter=50,
            nredo=1,
            verbose=False,
            seed=seed + _,
            spherical=True,
            gpu=False,
            min_points_per_centroid=5,
        )
        kmeans.centroids = init_centroids.astype('float32')
        kmeans.train(embeddings)

        # Assign all points to clusters
        _, I = kmeans.index.search(x=embeddings, k=1)
        labels = I.flatten()

        # Compute medoid as the point with minimal sum of distances in its cluster
        medoid_candidates = []
        for i in range(n_clusters):
            cluster_points = embeddings[labels == i]
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_points) == 0:
                continue
            # Compute sum of distances to all other points in cluster
            distances = ((cluster_points - cluster_points[:, None]) ** 2).sum(-1)
            sum_dist = distances.sum(axis=1)
            medoid_idx_in_cluster = np.argmin(sum_dist)
            medoid_candidates.append(cluster_indices[medoid_idx_in_cluster])

        inertia = kmeans.obj[-1]  # final objective
        if inertia < best_inertia:
            best_inertia = inertia
            best_medoids = np.array(medoid_candidates)

    return best_medoids