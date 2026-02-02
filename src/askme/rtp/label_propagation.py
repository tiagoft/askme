from typing import Tuple

import faiss
import numpy as np
from scipy.sparse import csr_matrix

class LabelPropagation:
    def __init__(
        self,
        faiss_index: faiss.Index,
        n_neighbors: int = 10,
        sigma: float = 1.0,
        alpha: float = 0.99,
        max_iter: int = 100,
        tol: float = 1e-3,
    ):
        self.faiss_index = faiss_index
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Step 1: Build k-NN graph
        indices, distances = make_knn_graph(
            X,
            index=self.faiss_index,
            n_neighbors=self.n_neighbors,
        )

        # Step 2: Build sparse affinity matrix W
        W = sparse_affinity(
            indices,
            distances,
            sigma=self.sigma,
        )

        # Step 3: Propagate labels
        y_pred = propagate_labels(
            W,
            y,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        return y_pred

# Step 1: Build k-NN graph using FAISS
def make_knn_graph(
    X : np.array,
    index: faiss.Index,
    n_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a k-nearest neighbors graph using a pre-built and populated FAISS index.
    
    The function performs a range search or exact k-NN search on the vectors already
    stored in the index (i.e., queries the index against itself).
    
    Parameters
    ----------
    index : faiss.Index
        A trained and populated FAISS index (e.g., IndexFlatL2, IndexIVFFlat on GPU/CPU, HNSW, etc.).
        Must have vectors already added via index.add().
    n_neighbors : int
        Number of neighbors to retrieve per vector (excluding the vector itself).
    
    Returns
    -------
    indices : np.ndarray
        Array of shape (n_vectors, n_neighbors) containing the indices of the nearest neighbors.
        dtype: int64
    distances : np.ndarray
        Array of shape (n_vectors, n_neighbors) containing the corresponding squared distances
        (or actual distances depending on the metric).
        dtype: float32
    
    Notes
    -----
    - The index must contain the vectors you want to build the graph over.
    - We search for (n_neighbors + 1) to include the self-match (distance 0), then drop it.
    - Works with both CPU and GPU indices.
    """
    # Get number of vectors in the index
    n_vectors = index.ntotal
    if n_vectors == 0:
        raise ValueError("FAISS index is empty. Add vectors with index.add() first.")

    if n_neighbors >= n_vectors:
        raise ValueError(f"n_neighbors ({n_neighbors}) must be < number of vectors ({n_vectors}).")

    # Search for one extra neighbor to include and then exclude the self-match
    k_search = n_neighbors + 1

    # Direct search: query the index with the same vectors it contains
    distances, indices = index.search(X, k_search)

    # Remove the first neighbor (which is always the point itself, distance ≈ 0)
    distances = 1 - distances[:, 1:]   # shape: (n_vectors, n_neighbors)
    indices = indices[:, 1:]       # shape: (n_vectors, n_neighbors)

    return indices, distances


# Step 2: Build sparse symmetric affinity matrix W with RBF kernel
def sparse_affinity(
    indices: np.ndarray,
    distances: np.ndarray,
    sigma: float = 1.0,
) -> csr_matrix:
    """
    Computes affinity W_ij = exp(-||x_i - x_j||² / (2 σ²)) for neighbor pairs.
    W is made symmetric: W = (W + W^T)/2 (undirected graph).
    This is the standard Gaussian (RBF) similarity kernel.
    """
    n = indices.shape[0]
    row = np.repeat(np.arange(n), indices.shape[1])
    col = indices.ravel()
    data = np.exp(-distances.ravel()**2 / (2 * sigma**2))
    W = csr_matrix((data, (row, col)), shape=(n, n))
    W = (W + W.T) / 2  # Symmetrize
    return W


# Step 3: Iterative Label Spreading (core propagation)
def propagate_labels(
    W: csr_matrix,
    y: np.ndarray,
    alpha: float = 0.99,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Implements the iterative update from Zhou et al. (2004):
        F^{t+1} = α S F^t + (1-α) Y
    with hard clamping of labeled points after each update.
    
    Parameters:
        W: sparse affinity matrix (n x n)
        y: label vector (-1 for unlabeled, class index otherwise)
        alpha: clamping factor (0 < alpha < 1). Higher alpha = stronger propagation.
        max_iter/tol: convergence criteria
    
    Returns:
        predicted class indices
    """
    n = W.shape[0]
    labels = np.unique(y[y >= 0])  # Known class indices
    num_classes = len(labels)

    # Initial soft label matrix Y (one-hot for labeled, zero for unlabeled)
    Y = np.zeros((n, num_classes))
    for i, lbl in enumerate(labels):
        Y[y == lbl, i] = 1.0

    clamp_mask = (y >= 0)  # Boolean mask for labeled points

    F = Y.copy()  # Current soft labels (probabilities)

    # Row-normalized propagation matrix S = D^{-1} W
    # (equivalent to random-walk Laplacian normalization)
    D = np.array(W.sum(axis=1)).flatten()  # Degree vector
    D_inv = csr_matrix((1 / (D + 1e-12), (np.arange(n), np.arange(n))),
                       shape=(n, n))  # Avoid div-by-0
    S = D_inv.dot(W)

    # Iterative propagation
    for iter_ in range(max_iter):
        F_new = alpha * S.dot(F) + (
            1 - alpha) * Y  # Propagation + pull toward initial labels

        # Hard clamping: force labeled points back to their original one-hot labels
        F_new[clamp_mask] = Y[clamp_mask]

        # Check convergence (max change in soft labels)
        if np.abs(F_new - F).max() < tol:
            print(f"Converged after {iter_+1} iterations")
            break

        F = F_new
    else:
        print(f"Reached max_iter ({max_iter}) without full convergence")

    # Final hard predictions
    return labels[np.argmax(F, axis=1)]
