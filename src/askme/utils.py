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
        niter=50,
        verbose=True,
        gpu=False,
    )
    kmeans.train(X, init_centroids=X[:n_clusters,:])

    # medoid indexes
    # Find medoids
    distances_to_centroids, _ = kmeans.index.search(X, k=n_clusters)
    #print(distances_to_centroids.shape, distances_to_centroids)
    medoid_indices = np.argmin(distances_to_centroids, axis=0)
    return medoid_indices
