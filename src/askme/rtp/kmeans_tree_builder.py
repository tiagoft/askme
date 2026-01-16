"""KMeansTreeBuilder class for building trees using K-means clustering."""

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import faiss
import numpy as np
from tqdm import tqdm

from askme.utils import NLIWithChunkingAndPooling

from ..askquestions import check_entailment, models
from ..makequestions import api, makequestion
from ..utils import (TextEmbeddingWithChunker, kmeans_with_faiss,
                     select_n_random_indices, vote_k_sampling)
from .label_propagation import (make_knn_graph, propagate_labels,
                                sparse_affinity)
from .make_collection_index import make_faiss_index
from .tree_models import SplitMetrics, TreeNode


class KMeansTreeBuilder:
    """
    KMeansTreeBuilder class that uses K-means clustering (k=2) to build binary trees.
    
    This class separates documents into two clusters using k-means, then asks an LLM
    to generate a hypothesis that distinguishes between the clusters.
    
    The algorithm works as follows:
    1. Gets embeddings for all documents
    2. Runs k-means with k=2 for those documents
    3. Asks LLM what is a hypothesis that is true for elements in "cluster 1", 
       but is false for elements in "cluster 2"
    4. Uses NLI to validate the hypothesis
    5. Uses label propagation to assign all documents
    
    Attributes:
        embedding_model: Text embedding model for vectorizing documents
        nli_model: NLI model for checking entailment
        nli_tokenizer: Tokenizer for the NLI model
        llm_model: Language model for generating hypotheses
        gpu_resources: FAISS GPU resources (None if use_gpu is False)
        use_gpu: Whether to use GPU acceleration
        embedding_model_name: Name of the embedding model
        nli_model_name: Name of the NLI model
        llm_model_name: Name of the LLM model
        chunk_size: Size of text chunks for embedding
        overlap: Overlap between chunks
        n_medoids_per_cluster: Number of medoids to select from each cluster
        n_documents_to_answer: Number of documents to label with NLI
        knn_neighbors: Number of neighbors for k-NN graph
        alpha: Alpha parameter for label propagation
        max_iter: Maximum iterations for label propagation
        tol: Tolerance for label propagation convergence
        max_retries: Maximum number of retries if split ratio is bad
        min_split_ratio: Minimum acceptable split ratio (None if no check)
        max_split_ratio: Maximum acceptable split ratio (None if no check)
        verbose: Whether to print verbose output during execution
    """

    def __init__(
        self,
        use_gpu: bool = False,
        embedding_model_name:
        str = 'sentence-transformers/paraphrase-albert-small-v2',
        nli_model_name:
        str = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
        llm_model_name: str = "gpt-4o-mini",
        chunk_size: int = 50,
        overlap: int = 10,
        n_medoids_per_cluster: int = 2,
        n_documents_to_answer: Union[int, str, float] = 'all',
        knn_neighbors: int = 2,
        nli_batch_size: int = 16,
        alpha: float = 0.99,
        max_iter: int = 100,
        tol: float = 1e-3,
        max_retries: int = 3,
        min_split_ratio: Optional[float] = None,
        max_split_ratio: Optional[float] = None,
        verbose: bool = False,
        cache_dir: str | None = None,
        selection_strategy: str = 'kmeans',
        nli_selection_strategy: str = 'kmeans',
        nli_batched: bool = True,
    ):
        """
        Initialize the KMeansTreeBuilder with all necessary models.
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: False)
            embedding_model_name: Name of the sentence transformer model
            nli_model_name: Name of the NLI model
            llm_model_name: Name of the LLM model for hypothesis generation
            chunk_size: Size of text chunks in words
            overlap: Overlap between chunks in words
            n_medoids_per_cluster: Number of medoids to select from each cluster
            n_documents_to_answer: Number of documents to label with NLI
            knn_neighbors: Number of neighbors for k-NN graph
            nli_batch_size: Batch size for NLI model
            alpha: Alpha parameter for label propagation (0 < alpha < 1)
            max_iter: Maximum iterations for label propagation
            tol: Tolerance for label propagation convergence
            max_retries: Maximum number of retries if split ratio is bad (default: 3)
            min_split_ratio: Minimum acceptable split ratio (proportion in smaller child).
                           If None, no minimum check is performed.
            max_split_ratio: Maximum acceptable split ratio (proportion in smaller child).
                           If None, no maximum check is performed.
            verbose: Whether to print verbose output during execution
            cache_dir: Directory to cache embeddings (default: None)
            selection_strategy: Strategy for selecting medoids ('kmeans', 'random', 'votek')
            nli_selection_strategy: Strategy for selecting documents to label ('kmeans', 'random', 'votek')
            nli_batched: Whether to use batched NLI calls (default: True)
        """
        self.use_gpu = use_gpu
        self.embedding_model_name = embedding_model_name
        self.nli_model_name = nli_model_name
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n_medoids_per_cluster = n_medoids_per_cluster
        self.n_documents_to_answer = n_documents_to_answer
        self.knn_neighbors = knn_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.max_retries = max_retries
        self.min_split_ratio = min_split_ratio
        self.max_split_ratio = max_split_ratio
        self.verbose = verbose
        self.selection_strategy = selection_strategy
        self.nli_batch_size = nli_batch_size
        self.nli_batched = nli_batched
        self.nli_selection_strategy = nli_selection_strategy

        # Determine device
        device = 'cuda' if use_gpu else 'cpu'

        # Initialize GPU resources for FAISS if needed
        self.gpu_resources = None
        if use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

        # Initialize embedding model
        self.embedding_model = TextEmbeddingWithChunker(
            model_name=embedding_model_name,
            chunk_size=chunk_size,
            overlap=overlap,
            device=device,
        )

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).expanduser()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        if self.cache_dir is not None:
            self.embedding_model.load_cache(
                str(self.cache_dir / 'embedding_cache.pkl'))
            if verbose:
                print(
                    f"Loaded embedding cache from {self.cache_dir / 'embedding_cache.pkl'}"
                )
                print(
                    f"Cache contains {len(self.embedding_model.cache)} entries."
                )

        # Initialize NLI model
        self.nli_model, self.nli_tokenizer = models.make_nli_model(
            model_name=nli_model_name)

        if use_gpu:
            self.nli_model = self.nli_model.to('cuda')

        if nli_batched:
            self.nli_batching_model = NLIWithChunkingAndPooling(
                nli_model=self.nli_model,
                tokenizer=self.nli_tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
                device=device,
                batch_size=nli_batch_size,
            )

        # Initialize LLM model
        if llm_model_name.startswith('gpt-4o'):
            self.llm_model = api.make_azure_model(llm_model_name)
        else:
            self.llm_model = api.make_ollama_model(llm_model_name)

    def __call__(
        self,
        X: Iterable[str],
        return_metrics: bool = False,
        initial_blacklist: Optional[list[str]] = None,
    ) -> Union[TreeNode, tuple[TreeNode, SplitMetrics]]:
        """
        Execute the K-means tree building algorithm on a collection of text documents.
        
        Args:
            X: Iterable of text documents (strings)
            return_metrics: If True, return (TreeNode, SplitMetrics), otherwise just TreeNode
            initial_blacklist: List of hypotheses to avoid
            
        Returns:
            TreeNode: Root node of the tree with document indices and question
            OR
            tuple[TreeNode, SplitMetrics]: TreeNode and metrics if return_metrics=True
        """
        start_time = time.time()
        metrics = SplitMetrics()

        # Convert iterable to list to allow indexing
        text_collection = list(X)

        if len(text_collection) == 0:
            raise ValueError("Text collection cannot be empty")

        # Step 1: Vectorize documents and create FAISS index
        faiss_start = time.time()
        if self.verbose:
            print("Vectorizing collection and creating FAISS index...")
        dimension = self.embedding_model.model.get_sentence_embedding_dimension()
        faiss_index, embeddings = make_faiss_index(
            text_collection=text_collection,
            embedding_model=self.embedding_model,
            dimension=dimension,
            use_gpu=self.use_gpu,
            return_embeddings=True,
            gpu_resources=self.gpu_resources,
        )
        if self.cache_dir is not None:
            self.embedding_model.save_cache(
                str(self.cache_dir / 'embedding_cache.pkl'))
        metrics.faiss_search_time_ms = (time.time() - faiss_start) * 1000

        # Step 2: Run k-means with k=2 to split documents into 2 clusters
        kmeans_start = time.time()
        if self.verbose:
            print("Running k-means with k=2 to split documents...")
        
        # Run k-means to get 2 clusters
        kmeans = faiss.Kmeans(
            d=faiss_index.d,
            k=2,
            niter=50,
            nredo=5,
            verbose=False,
            seed=42,
            spherical=True,
            gpu=self.use_gpu,
        )
        kmeans.train(embeddings)
        
        # Assign each document to a cluster
        _, cluster_assignments = kmeans.index.search(embeddings, 1)
        cluster_assignments = cluster_assignments.flatten()
        
        # Get document indices for each cluster
        cluster_0_docs = [i for i, c in enumerate(cluster_assignments) if c == 0]
        cluster_1_docs = [i for i, c in enumerate(cluster_assignments) if c == 1]
        
        metrics.kmeans_time_ms = (time.time() - kmeans_start) * 1000
        
        if self.verbose:
            print(f"K-means split: Cluster 0 has {len(cluster_0_docs)} docs, Cluster 1 has {len(cluster_1_docs)} docs")

        # Handle edge case where one cluster is empty
        if len(cluster_0_docs) == 0 or len(cluster_1_docs) == 0:
            if self.verbose:
                print("One cluster is empty, returning leaf node.")
            root = TreeNode(
                documents=list(range(len(text_collection))),
                question=None,
            )
            if return_metrics:
                metrics.success = False
                return root, metrics
            return root

        # Step 3: Select medoids from each cluster for LLM prompt
        medoid_start = time.time()
        if self.verbose:
            print(f"Selecting {self.n_medoids_per_cluster} medoids from each cluster...")
        
        # Select medoids from cluster 0
        cluster_0_embeddings = embeddings[cluster_0_docs]
        n_medoids_0 = min(self.n_medoids_per_cluster, len(cluster_0_docs))
        if self.selection_strategy == 'random':
            local_indices_0 = select_n_random_indices(
                total_size=len(cluster_0_docs),
                n_select=n_medoids_0,
                seed=1234,
            )
        elif self.selection_strategy == 'kmeans':
            # Create a temporary index for cluster 0
            temp_index_0 = faiss.IndexFlatL2(dimension)
            temp_index_0.add(cluster_0_embeddings)
            local_indices_0 = kmeans_with_faiss(
                faiss_index=temp_index_0,
                X=cluster_0_embeddings,
                n_clusters=n_medoids_0,
                use_gpu=False,  # Use CPU for small clusters
            )
        elif self.selection_strategy == 'votek':
            temp_index_0 = faiss.IndexFlatL2(dimension)
            temp_index_0.add(cluster_0_embeddings)
            local_indices_0 = vote_k_sampling(
                temp_index_0,
                cluster_0_embeddings,
                n_clusters=n_medoids_0,
                k_neighbors=min(30, len(cluster_0_docs) - 1),
            )
        
        medoid_indices_0 = [cluster_0_docs[i] for i in local_indices_0]
        medoids_0 = [text_collection[i] for i in medoid_indices_0]

        # Select medoids from cluster 1
        cluster_1_embeddings = embeddings[cluster_1_docs]
        n_medoids_1 = min(self.n_medoids_per_cluster, len(cluster_1_docs))
        if self.selection_strategy == 'random':
            local_indices_1 = select_n_random_indices(
                total_size=len(cluster_1_docs),
                n_select=n_medoids_1,
                seed=1235,
            )
        elif self.selection_strategy == 'kmeans':
            temp_index_1 = faiss.IndexFlatL2(dimension)
            temp_index_1.add(cluster_1_embeddings)
            local_indices_1 = kmeans_with_faiss(
                faiss_index=temp_index_1,
                X=cluster_1_embeddings,
                n_clusters=n_medoids_1,
                use_gpu=False,
            )
        elif self.selection_strategy == 'votek':
            temp_index_1 = faiss.IndexFlatL2(dimension)
            temp_index_1.add(cluster_1_embeddings)
            local_indices_1 = vote_k_sampling(
                temp_index_1,
                cluster_1_embeddings,
                n_clusters=n_medoids_1,
                k_neighbors=min(30, len(cluster_1_docs) - 1),
            )
        
        medoid_indices_1 = [cluster_1_docs[i] for i in local_indices_1]
        medoids_1 = [text_collection[i] for i in medoid_indices_1]

        metrics.medoid_selection_time_ms = (time.time() - medoid_start) * 1000

        if self.verbose:
            print(f"Selected {len(medoids_0)} medoids from cluster 0: {medoid_indices_0}")
            print(f"Selected {len(medoids_1)} medoids from cluster 1: {medoid_indices_1}")

        # Retry loop for generating a good question
        if initial_blacklist is not None:
            blacklist = initial_blacklist
        else:
            blacklist = []

        hypothesis = None
        propagated_labels = None
        left_docs = []
        right_docs = []

        for attempt in range(self.max_retries + 1):
            if self.verbose:
                print(f"Attempt {attempt + 1} to generate hypothesis and split documents...")

            # Step 4: Generate hypothesis using LLM
            # The hypothesis should be true for cluster 1, false for cluster 0
            llm_start = time.time()
            try:
                # Combine medoids from both clusters
                # We present cluster 1 medoids first (these should satisfy the hypothesis)
                combined_medoids = medoids_1 + medoids_0
                response = makequestion.make_a_question_about_collection(
                    collection=combined_medoids,
                    model=self.llm_model,
                    retries=15,
                    blacklist=blacklist,
                )
            except Exception as e:
                print(f"LLM call failed during hypothesis generation: {e}")
                print("Returning leaf node with all documents.")
                root = TreeNode(
                    documents=list(range(len(text_collection))),
                    question=None,
                )
                if return_metrics:
                    metrics.success = False
                    return root, metrics
                return root

            # Track LLM tokens
            metrics.llm_input_tokens += response.usage().input_tokens
            metrics.llm_output_tokens += response.usage().output_tokens
            hypothesis = response.output.hypothesis
            metrics.llm_request_time_ms += (time.time() - llm_start) * 1000
            if self.verbose:
                print(f"Generated hypothesis: {hypothesis}")

            # Step 5: Use NLI to validate the hypothesis
            # We expect cluster 1 documents to satisfy the hypothesis (label=1)
            # and cluster 0 documents to not satisfy it (label=0)
            
            # Determine which documents to label with NLI
            nli_start = time.time()
            if self.n_documents_to_answer == 'all':
                doc_indices = list(range(len(text_collection)))
                n_docs_to_label = len(text_collection)
            elif self.n_documents_to_answer == 'same':
                # Use the medoids we already selected
                doc_indices = medoid_indices_0 + medoid_indices_1
                n_docs_to_label = len(doc_indices)
            else:
                if isinstance(self.n_documents_to_answer, float):
                    n_docs_to_label = int(self.n_documents_to_answer * len(text_collection))
                else:
                    n_docs_to_label = int(self.n_documents_to_answer)
                n_docs_to_label = min(n_docs_to_label, len(text_collection))
                
                if self.nli_selection_strategy == 'random':
                    doc_indices = select_n_random_indices(
                        total_size=len(text_collection),
                        n_select=n_docs_to_label,
                        seed=1234 + attempt,
                    )
                elif self.nli_selection_strategy == 'votek':
                    doc_indices = vote_k_sampling(
                        faiss_index,
                        embeddings,
                        n_clusters=n_docs_to_label,
                        k_neighbors=30,
                    )
                else:  # Default to kmeans
                    doc_indices = kmeans_with_faiss(
                        faiss_index=faiss_index,
                        X=embeddings,
                        n_clusters=n_docs_to_label,
                        use_gpu=self.use_gpu,
                    )

            # Initialize labels as unlabeled (-1)
            answers = -np.ones((len(text_collection),), dtype=np.int8)

            device = 'cuda' if self.use_gpu else 'cpu'
            nli_confidences = []

            # Use NLI to label selected documents
            if self.nli_batched is False:
                for doc_index in tqdm(doc_indices):
                    document = text_collection[doc_index]
                    pooled_results = check_entailment.pool_nli_scores(
                        check_fn=check_entailment.check_entailment_nli,
                        premise=document,
                        hypothesis=hypothesis,
                        chunk_size=200,
                        overlap=20,
                        model=self.nli_model,
                        tokenizer=self.nli_tokenizer,
                        device=device,
                    )
                    entails, entailment_score, contradiction_score, P_entailment = pooled_results
                    answers[doc_index] = 1 if entails else 0
                    nli_confidences.append(P_entailment)
                    metrics.nli_calls += 1
            else:
                if self.verbose:
                    print("Using batched NLI calls...")
                
                premises = [text_collection[doc_index] for doc_index in doc_indices]
                batched_results = self.nli_batching_model(
                    premise=premises,
                    hypothesis=hypothesis,
                )
                
                for i, doc_index in enumerate(doc_indices):
                    entails, entailment_score, contradiction_score, P_entailment = batched_results[i]
                    answers[doc_index] = 1 if entails else 0
                    nli_confidences.append(P_entailment)
                    metrics.nli_calls += 1

            metrics.nli_time_ms += (time.time() - nli_start) * 1000

            if self.verbose:
                print(f"NLI answered {len(doc_indices)} documents.")
                print(f"Number of 1s: {np.sum(answers == 1)}, Number of 0s: {np.sum(answers == 0)}")

            # Calculate average NLI confidence
            if nli_confidences:
                metrics.medoid_nli_confidence_avg = sum(nli_confidences) / len(nli_confidences)

            # Step 6: Propagate labels
            label_prop_start = time.time()
            indices, distances = make_knn_graph(
                np.array(embeddings).astype('float32'),
                faiss_index,
                n_neighbors=self.knn_neighbors)
            sigma = np.sqrt(np.median(distances[:, -1]))
            W = sparse_affinity(indices, distances, sigma=sigma)
            propagated_labels = propagate_labels(W,
                                                 answers,
                                                 alpha=self.alpha,
                                                 max_iter=self.max_iter,
                                                 tol=self.tol)
            metrics.label_propagation_time_ms += (time.time() - label_prop_start) * 1000

            if self.verbose:
                n_labeled = np.sum(answers != -1)
                n_propagated_1 = np.sum(propagated_labels == 1)
                n_propagated_0 = np.sum(propagated_labels == 0)
                print(f"Label propagation completed. Initially labeled: {n_labeled}, "
                      f"Propagated 1s: {n_propagated_1}, Propagated 0s: {n_propagated_0}")

            # Step 7: Build TreeNode based on propagated labels
            # Documents where label == 1 go to left child, label == 0 go to right child
            left_docs = [i for i, label in enumerate(propagated_labels) if label == 1]
            right_docs = [i for i, label in enumerate(propagated_labels) if label == 0]

            # Check split ratio if split occurred
            split_is_valid = True
            metrics.n_attempts = attempt + 1
            if len(left_docs) > 0 and len(right_docs) > 0:
                # Calculate split ratio (proportion in smaller child)
                smaller_count = min(len(left_docs), len(right_docs))
                split_ratio = smaller_count / len(text_collection)

                # Check if split ratio is within acceptable range
                if self.min_split_ratio is not None and split_ratio < self.min_split_ratio:
                    split_is_valid = False
                if self.max_split_ratio is not None and split_ratio > self.max_split_ratio:
                    split_is_valid = False

                # Store split ratio and split entropy
                if attempt == self.max_retries or split_is_valid:
                    metrics.split_ratio = len(left_docs) / len(text_collection)
                    entropy = -(split_ratio * np.log2(split_ratio) +
                               (1 - split_ratio) * np.log2(1 - split_ratio))
                    metrics.split_entropy = entropy

            # If split is valid or we've exhausted retries, stop
            if split_is_valid or attempt == self.max_retries:
                if self.verbose:
                    print("Accepting hypothesis and split.")
                    print(f"Split ratio: {len(left_docs) / len(text_collection):.3f}")
                    print(f"Left documents: {len(left_docs)}, Right documents: {len(right_docs)}")
                    print(f"Retries: {attempt}")
                break

            # Otherwise, add this hypothesis to the blacklist and retry
            blacklist.append(hypothesis)

            if self.verbose:
                print(f"Hypothesis '{hypothesis}' led to invalid split.")
                print(f"Split ratio {split_ratio:.3f} not acceptable. Retrying...")

        if self.verbose:
            print(f"Final hypothesis: {hypothesis}")
            print(f"Left documents: {len(left_docs)}, Right documents: {len(right_docs)}")

        # Create tree node
        root = TreeNode(
            documents=list(range(len(text_collection))),
            question=hypothesis,
        )

        # Only create children if there's a meaningful split
        if len(left_docs) > 0 and len(right_docs) > 0:
            root.left = TreeNode(documents=left_docs)
            root.right = TreeNode(documents=right_docs)

        # Calculate total time
        metrics.total_time_ms = (time.time() - start_time) * 1000
        root.blacklist = blacklist

        if self.verbose:
            print(f"KMeansTreeBuilder execution completed in {metrics.total_time_ms:.2f} ms")

        if return_metrics:
            return root, metrics
        return root


class KMeansTreeRecursion:
    """
    KMeansTreeRecursion class that recursively builds K-means trees with stopping criteria.
    
    This class wraps a KMeansTreeBuilder and recursively applies it to build a complete
    tree structure with configurable stopping criteria.
    
    Attributes:
        builder: Pre-initialized KMeansTreeBuilder instance
        min_node_size: Minimum number of documents for a node to be split
        min_split_ratio: Minimum split ratio (proportion in smaller child)
        max_split_ratio: Maximum split ratio (proportion in smaller child)
        max_depth: Maximum depth of the tree
    """

    def __init__(
        self,
        builder: KMeansTreeBuilder,
        min_node_size: int = 2,
        min_split_ratio: float = 0.2,
        max_split_ratio: float = 0.8,
        max_depth: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize KMeansTreeRecursion with a builder and stopping criteria.
        
        Args:
            builder: Pre-initialized KMeansTreeBuilder instance
            min_node_size: Minimum number of documents required to split a node
            min_split_ratio: Minimum split ratio for a valid split (default: 0.2)
            max_split_ratio: Maximum split ratio for a valid split (default: 0.8)
            max_depth: Maximum depth of the tree (default: 10)
            verbose: Whether to print verbose output during execution
        """
        self.builder = builder
        self.min_node_size = min_node_size
        self.min_split_ratio = min_split_ratio
        self.max_split_ratio = max_split_ratio
        self.max_depth = max_depth
        self.verbose = verbose

    def __call__(self, X: Iterable[str]) -> tuple[TreeNode, SplitMetrics]:
        """
        Recursively build a K-means tree from a collection of text documents.
        
        Args:
            X: Iterable of text documents (strings)
            
        Returns:
            tuple[TreeNode, SplitMetrics]: Root node of the tree and global metrics
        """
        if self.verbose:
            print("Starting KMeansTreeRecursion...")
        
        # Convert to list and store original text collection
        text_collection = list(X)

        # Build the tree recursively and accumulate metrics
        root, global_metrics = self._recurse(
            text_collection=text_collection,
            document_indices=list(range(len(text_collection))),
            depth=0,
        )

        return root, global_metrics

    def _recurse(
        self,
        text_collection: list[str],
        document_indices: list[int],
        depth: int,
        current_blacklist: Optional[list[str]] = None,
    ) -> tuple[TreeNode, SplitMetrics]:
        """
        Recursively build the tree structure.
        
        Args:
            text_collection: The full text collection (for reference)
            document_indices: Original indices of documents in this node
            depth: Current depth in the tree
            current_blacklist: List of hypotheses to avoid
            
        Returns:
            tuple[TreeNode, SplitMetrics]: The node for this subset and accumulated metrics
        """
        if self.verbose:
            print(f"Entering recursion at depth {depth} with {len(document_indices)} documents.")
        
        # Get the subset of documents for this node
        node_documents = [text_collection[i] for i in document_indices]

        # Check stopping criteria
        should_stop = (len(document_indices) < self.min_node_size
                       or depth >= self.max_depth)

        if should_stop:
            # Create leaf node
            return TreeNode(documents=document_indices), SplitMetrics()

        # Execute KMeansTreeBuilder for current node
        node_root, node_metrics = self.builder(
            node_documents,
            return_metrics=True,
            initial_blacklist=current_blacklist)

        # Create tree node with original document indices
        node_root.metrics = node_metrics

        # Check if split is valid
        if node_root.left is None or node_root.right is None:
            # No split occurred
            return node_root, node_metrics

        # Check split ratio criteria
        left_count = len(node_root.left.documents)
        right_count = len(node_root.right.documents)
        total_count = left_count + right_count

        if total_count == 0:
            return node_root, node_metrics

        # Calculate split ratio (proportion in smaller child)
        smaller_count = min(left_count, right_count)
        split_ratio = smaller_count / total_count

        # Check if split ratio is within acceptable range
        if split_ratio < self.min_split_ratio or split_ratio > self.max_split_ratio:
            # Split ratio not acceptable, don't recurse
            return node_root, node_metrics

        # Map local indices back to original indices
        left_original_indices = [document_indices[i] for i in node_root.left.documents]
        right_original_indices = [document_indices[i] for i in node_root.right.documents]

        # Recurse into children and accumulate metrics
        node_root.left, left_metrics = self._recurse(
            text_collection=text_collection,
            document_indices=left_original_indices,
            depth=depth + 1,
            current_blacklist=node_root.blacklist,
        )

        node_root.right, right_metrics = self._recurse(
            text_collection=text_collection,
            document_indices=right_original_indices,
            depth=depth + 1,
            current_blacklist=node_root.blacklist,
        )

        # Combine metrics from current node and children
        combined_metrics = node_metrics + left_metrics + right_metrics

        return node_root, combined_metrics
