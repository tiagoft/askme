from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from collections import defaultdict

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
        self.cache = {}

    def __call__(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        chunk_embeddings = self.model.encode(chunks)
        embedding = self.pooling_fn(chunk_embeddings, axis=0)
        self.cache[text] = embedding
        return embedding

    def save_cache(self, path: str, append: bool = False):
        if append:
            try:
                with open(path, 'rb') as f:
                    current_cache = pickle.load(f)
                self.cache.update(current_cache)
            except FileNotFoundError:
                pass
        with open(path, 'wb') as f:
            pickle.dump(self.cache, f)

    def load_cache(self, path: str):
        try:
            with open(path, 'rb') as f:
                self.cache = pickle.load(f)
        except FileNotFoundError:
            return {}

def select_n_random_indices(
    total_size: int,
    n_select: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Select n unique random indices from a range of total_size.
    
    Args:
        total_size: The total number of items
        n_select: Number of unique indices to select
        seed: Random seed for reproducibility
    Returns:
        Numpy array of selected indices
    """
    np.random.seed(seed)
    indices = np.random.choice(total_size, size=n_select, replace=False)
    return indices

def fast_votek(embeddings,select_num,k,vote_file=None):
    # This code was adapted from 
    # https://github.com/xlang-ai/icl-selective-annotation/blob/main/two_steps.py#L99
    # (I changed the call to cosine similarity to a call to euclidean distances for testing;
    # also I used -distance instead of distance).
    # This allowed to run this and the faiss-version of votek in the same demo.
    # Check demo_selection.py for usage.
    
    
    from tqdm import tqdm
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from collections import defaultdict
    import json
    
    n = len(embeddings)
    if vote_file is not None and os.path.isfile(vote_file):
        with open(vote_file) as f:
            vote_stat = json.load(f)
    else:
        bar = tqdm(range(n),desc=f'voting')
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(-euclidean_distances(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
            for idx in sorted_indices:
                if idx!=i:
                    vote_stat[idx].append(i)
            bar.update(1)
        if vote_file is not None:
            with open(vote_file,'w') as f:
                json.dump(vote_stat,f)
    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices)<select_num:
        cur_scores = defaultdict(int)
        for idx,candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices


def vote_k_sampling(
    faiss_index: faiss.Index,
    X: np.ndarray,
    n_clusters: int,
    k_neighbors: int = 15,
) -> np.ndarray:

    distances, indices = faiss_index.search(X, k_neighbors + 1)
    
    vote_stat = defaultdict(list)
    for i in range(X.shape[0]):
        for neighbor_idx in indices[i, 1:]: # Skip self
            vote_stat[neighbor_idx].append(i)
    
    # 2. Selection Loop (The original Su et al. logic)
    selected_indices = []
    selected_times = defaultdict(int)
    
    # Convert vote_stat to a list of tuples for sorting
    votes_list = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
    
    while len(selected_indices) < n_clusters:
        cur_scores = defaultdict(int)
        for idx, supporters in votes_list:
            if idx in selected_indices:
                continue
            for supporter in supporters:
                cur_scores[idx] += 10 ** (-selected_times[supporter])
        
        if not cur_scores:
            break
        
        best_idx = max(cur_scores, key=cur_scores.get)
        selected_indices.append(int(best_idx))
        
        for supporter in vote_stat[best_idx]:
            selected_times[supporter] += 1
            
    return selected_indices



def kmeans_with_faiss(
    faiss_index: faiss.Index,
    X: np.ndarray,
    n_clusters: int,
    use_gpu: bool = True,
    seed: int = 42,
    niter: int = 50,
    spherical: bool = True,
) -> np.ndarray:

    kmeans = faiss.Kmeans(
        d=faiss_index.d,
        k=n_clusters,
        niter=niter,
        nredo=5,
        verbose=False,
        seed=seed,
        spherical=spherical,
        gpu=use_gpu,
        
    )
    kmeans.train(X)  #, init_centroids=X[:n_clusters,:])
    centroids = kmeans.centroids
    D, I = faiss_index.search(centroids,
                              1)  # D: distances, I: indices (shape k, 1)

    medoid_indices = I.flatten()  # Indices of medoids in your original data
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
        seed: int = 1234,
        max_docs: int = None) -> np.ndarray:
    """
    Returns indices of K unique medoids using a robust k-medoids approximation.
    """
    if max_docs is not None and embeddings.shape[0] > max_docs:
        np.random.seed(seed)
        sampled_indices = np.random.choice(
            embeddings.shape[0], max_docs, replace=False)
        embeddings = embeddings[sampled_indices]
        
    d = embeddings.shape[1]
    best_medoids = None
    best_inertia = np.inf

    for _ in range(nredo):
        # Randomly sample initial medoids
        np.random.seed(seed + _)
        init_indices = np.random.choice(len(embeddings),
                                        n_clusters,
                                        replace=False)
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
            distances = ((cluster_points - cluster_points[:, None])**2).sum(-1)
            sum_dist = distances.sum(axis=1)
            medoid_idx_in_cluster = np.argmin(sum_dist)
            medoid_candidates.append(cluster_indices[medoid_idx_in_cluster])

        inertia = kmeans.obj[-1]  # final objective
        if inertia < best_inertia:
            best_inertia = inertia
            best_medoids = np.array(medoid_candidates)

    return best_medoids
