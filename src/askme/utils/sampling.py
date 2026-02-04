import numpy as np
import faiss
from collections import defaultdict
from askme.config.config import SamplingConfig, config_factory





class Sampler:

    def __init__(self):
        raise NotImplementedError

    def __call__(self, faiss_index: faiss.Index | None = None, X: np.ndarray | None = None) -> np.ndarray:
        """Returns selected indices as a numpy array."""
        raise NotImplementedError

def sampler_factory(
    config: SamplingConfig = config_factory(SamplingConfig),
) -> Sampler:
    """Factory function to create a sampler based on the sampler name."""
    sampler_name = config.selection_strategy
    if sampler_name == 'random':
        return RandomSampler(config)
    elif sampler_name == 'vote_k':
        return VoteKSampler(config)
    elif sampler_name == 'kmeans':
        return KMeansSampler(config)
    else:
        raise ValueError(f"Unknown sampler name: {sampler_name}")
    

class RandomSampler(Sampler):

    def __init__(
        self,
        config: SamplingConfig = config_factory(SamplingConfig),
    ):
        self.total_size = config.total_size
        self.n_select = config.n_select
        self.seed = config.seed

    def __call__(self, faiss_index: faiss.Index | None = None, X: np.ndarray | None = None,) -> np.ndarray:
        np.random.seed(self.seed)
        if X is not None:
            total_size = X.shape[0]
        else:
            total_size = self.total_size
        
        indices = np.random.choice(total_size,
                               size=self.n_select,
                               replace=False,)
        return indices

class VoteKSampler(Sampler):

    def __init__(
        self,
        config: SamplingConfig = config_factory(SamplingConfig),
    ):
        self.n_clusters = config.n_select
        self.k_neighbors = config.k_neighbors

    def __call__(self, faiss_index: faiss.Index,X: np.ndarray) -> np.ndarray:
        return vote_k_sampling(
            faiss_index,
            X,
            n_clusters=self.n_clusters,
            k_neighbors=self.k_neighbors,
        )


class KMeansSampler(Sampler):

    def __init__(
        self,
        
        config: SamplingConfig = config_factory(SamplingConfig)
    ):
        self.n_clusters = config.n_select
        self.use_gpu = config.use_gpu
        self.seed = config.seed
        self.niter = config.niter
        self.spherical = config.spherical

    def __call__(self, faiss_index: faiss.Index, X: np.ndarray) -> np.ndarray:
        return kmeans_with_faiss(
            faiss_index,
            X,
            n_clusters=self.n_clusters,
            use_gpu=self.use_gpu,
            seed=self.seed,
            niter=self.niter,
            spherical=self.spherical,
        )


def select_n_random_indices(
    total_size: int,
    n_select: int | float,
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
    if isinstance(n_select, float):
        n_select = int(total_size * n_select)
    indices = np.random.choice(total_size, size=n_select, replace=False)
    return indices


def fast_votek(embeddings, select_num, k, vote_file=None):
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
        bar = tqdm(range(n), desc=f'voting')
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(-euclidean_distances(embeddings, cur_emb),
                                axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k - 1:-1]
            for idx in sorted_indices:
                if idx != i:
                    vote_stat[idx].append(i)
            bar.update(1)
        if vote_file is not None:
            with open(vote_file, 'w') as f:
                json.dump(vote_stat, f)
    votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices) < select_num:
        cur_scores = defaultdict(int)
        for idx, candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:

                    cur_scores[idx] += 10**(-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices


def vote_k_sampling(
    faiss_index: faiss.Index,
    X: np.ndarray,
    n_clusters: int | float,
    k_neighbors: int = 15,
) -> np.ndarray:

    if isinstance(n_clusters, float):
        n_clusters = int(X.shape[0] * n_clusters)

    distances, indices = faiss_index.search(X, k_neighbors + 1)

    vote_stat = defaultdict(list)
    for i in range(X.shape[0]):
        for neighbor_idx in indices[i, 1:]:  # Skip self
            vote_stat[neighbor_idx].append(i)

    # 2. Selection Loop (The original Su et al. logic)
    selected_indices = []
    selected_times = defaultdict(int)

    # Convert vote_stat to a list of tuples for sorting
    votes_list = sorted(vote_stat.items(),
                        key=lambda x: len(x[1]),
                        reverse=True)

    while len(selected_indices) < n_clusters:
        cur_scores = defaultdict(int)
        for idx, supporters in votes_list:
            if idx in selected_indices:
                continue
            for supporter in supporters:
                cur_scores[idx] += 10**(-selected_times[supporter])

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
    n_clusters: int | float,
    use_gpu: bool = True,
    seed: int = 42,
    niter: int = 50,
    spherical: bool = True,
) -> np.ndarray:

    if isinstance(n_clusters, float):
        n_clusters = int(X.shape[0] * n_clusters)

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
        sampled_indices = np.random.choice(embeddings.shape[0],
                                           max_docs,
                                           replace=False)
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
