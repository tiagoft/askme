import numpy as np
import faiss
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import rbf_kernel  # Optional, for reference


# Step 1: Build approximate k-NN graph using FAISS
def build_knn_graph(X, k=20, use_gpu=True, gpu_id=0, nprobe=16):
    """
    Constructs a k-nearest-neighbor graph using FAISS, with optional GPU acceleration.
    
    Parameters:
        X: (n_samples, n_features) numpy array, float32
        k: number of neighbors (excluding self)
        use_gpu: bool, if True use GPU (if available), else CPU
        gpu_id: int, which GPU to use (default 0)
        nprobe: int, for IVF indices — number of clusters to search (higher = better accuracy, slower)
    
    Returns:
        indices: (n_samples, k) array of neighbor indices
        distances: (n_samples, k) array of squared L2 distances
    """
    X = np.ascontiguousarray(X.astype('float32'))
    n, d = X.shape

    # Search for k+1 to include self, then drop it later
    search_k = k + 1

    if use_gpu:
        # Configure GPU resources
        res = faiss.StandardGpuResources()
        # Optional: limit GPU memory if needed
        # res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory

        # For high accuracy + speed on GPU, use IVF Flat (recommended for >50k points)
        # You can also use IndexFlatL2 directly on GPU for exact search
        cpu_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d,
                                       max(1, int(np.sqrt(n))),
                                       faiss.METRIC_L2)
        
        # Train on CPU first (required for IVF)
        cpu_index.train(X)
        cpu_index.nprobe = nprobe  # Trade-off: accuracy vs speed

        # Move index to GPU
        index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)

        # Alternative: Exact search on GPU (fast for moderate dimensions)
        # index = faiss.GpuIndexFlatL2(res, d)

    else:
        # CPU version: use IVF for large datasets, or Flat for exact
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d,
                                   max(1, int(np.sqrt(n))), faiss.METRIC_L2)
        index.train(X)
        index.nprobe = nprobe

        # For exact search on CPU (slower but precise):
        # index = faiss.IndexFlatL2(d)

    # Add data
    index.add(X)

    # Search
    distances, indices = index.search(X, search_k)

    # Remove self-neighbors (first column is always the point itself)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    return indices, distances


# Step 2: Build sparse symmetric affinity matrix W with RBF kernel
def sparse_affinity(indices, distances, sigma=1.0):
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
def propagate_labels(W, y, alpha=0.99, max_iter=100, tol=1e-3):
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


# Example usage
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=50000,
                           n_features=1000,
                           n_classes=3,
                           n_informative=15,
                           random_state=42)
# Convert 90% of labels to -1 (unlabeled)
rng = np.random.default_rng(42)
mask = rng.uniform(size=y.shape) < 0.9
y[mask] = -1
import time

t0 = time.time()
indices, distances = build_knn_graph(
    X,
    k=20,
    use_gpu=True,
    gpu_id=0,
    nprobe=16,
)
t1 = time.time()
W = sparse_affinity(indices, distances,
                    sigma=np.median(distances))  # Good heuristic for sigma
t2 = time.time()
pred_y = propagate_labels(W, y, alpha=0.99)
t3 = time.time()
print(f"Time to build k-NN graph: {t1 - t0:.2f} seconds")
print(f"Time to build affinity matrix: {t2 - t1:.2f} seconds")
print(f"Time for label propagation: {t3 - t2:.2f} seconds")
# Evaluate accuracy on originally labeled points
true_labels = y[~mask]
pred_labels = pred_y[~mask]
accuracy = np.mean(true_labels == pred_labels)
print(f"Label propagation accuracy on labeled points: {accuracy * 100:.2f}%")
