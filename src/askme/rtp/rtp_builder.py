"""RTPBuilder class for operationalizing the RTP (Retrieval-based Tree Partitioning) algorithm."""

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import faiss
import numpy as np
from tqdm import tqdm

from askme.config.config import (MakeQuestionsConfig,
                                 NLIBatchingChukingConfig,
                                 SamplingConfig,
                                 TextEmbeddingConfig,
                                 LabelPropagationConfig,
                                 config_factory,)

from .nli import NLIWithChunkingAndPooling

from ..askquestions import check_entailment, models
from ..makequestions import api, makequestion
from ..utils import (TextEmbeddingWithChunker, kmeans_with_faiss,
                     select_n_random_indices, vote_k_sampling, sampler_factory)
from .label_propagation import (make_knn_graph, propagate_labels,
                                sparse_affinity, LabelPropagation)
from .make_collection_index import make_faiss_index
from .tree_models import SplitMetrics, TreeNode


class RTPBuilder:
    """
    RTPBuilder class that initializes models once and can be called multiple times
    to build RTP trees from text collections.
    
    This class separates the expensive initialization of models (embedding, NLI, LLM)
    from the execution of the RTP algorithm, allowing for efficient repeated calls.
    
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
        n_medoids: Number of medoids for hypothesis generation
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
        n_medoids: int = 4,
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
        selection_strategy: Union['kmeans', 'random', 'votek'] = 'kmeans',
        nli_selection_strategy: Union['kmeans', 'random', 'votek'] = 'kmeans',
        nli_batched: bool = True,
        embedding_model_config: TextEmbeddingConfig = config_factory(
            TextEmbeddingConfig),
        llm_model_config: MakeQuestionsConfig = config_factory(
            MakeQuestionsConfig),
        nli_config: NLIBatchingChukingConfig = config_factory(
            NLIBatchingChukingConfig),
        nli_sampler_config: SamplingConfig = config_factory(
            SamplingConfig,
            override_data='NLISamplingConfig',
        ),
        llm_sampler_config: SamplingConfig = config_factory(SamplingConfig,),
        label_propagation_config: LabelPropagationConfig = config_factory(
            LabelPropagationConfig, )
    ):
        """
        Initialize the RTPBuilder with all necessary models.
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: False)
            embedding_model_name: Name of the sentence transformer model
            nli_model_name: Name of the NLI model
            llm_model_name: Name of the LLM model for hypothesis generation
            chunk_size: Size of text chunks in words
            overlap: Overlap between chunks in words
            n_medoids: Number of medoids for hypothesis generation
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
            nli_batched: Whether to use batched NLI calls (default: True)
        """
        self.use_gpu = use_gpu
        self.max_retries = max_retries
        self.min_split_ratio = min_split_ratio
        self.max_split_ratio = max_split_ratio
        self.verbose = verbose
        self.nli_selection_strategy = nli_selection_strategy
        self.nli_sampler_config = nli_sampler_config
        self.llm_sampler_config = llm_sampler_config
        self.llm_model_config = llm_model_config
        self.embedding_model_config = embedding_model_config
        self.label_propagation_config = label_propagation_config

        # Determine device
        device = 'cuda' if use_gpu else 'cpu'

        # Initialize GPU resources for FAISS if needed
        self.gpu_resources = None
        if use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

        # Initialize embedding model
        self.embedding_model = TextEmbeddingWithChunker(
            config=embedding_model_config)

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
                    f"Cache contiains {len(self.embedding_model.cache)} entries."
                )

        self.nli_batching_model = NLIWithChunkingAndPooling(
            config=nli_config, )
        self.nli_config = nli_config

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
        Execute the RTP algorithm on a collection of text documents.
        
        Args:
            X: Iterable of text documents (strings)
            return_metrics: If True, return (TreeNode, SplitMetrics), otherwise just TreeNode
            
        Returns:
            TreeNode: Root node of the RTP tree with document indices and question
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
        dimension = self.embedding_model.model.get_sentence_embedding_dimension(
        )
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

        # Step 2: Get medoids via k-means for hypothesis generation
        t_initial_medoid = time.time()
        if self.verbose:
            print(
                f"Selecting {self.n_medoids} elements for hypothesis generation..."
            )
        llm_sampler = sampler_factory(self.llm_sampler_config)
        medoid_indices = llm_sampler(
            faiss_index=faiss_index,
            X=np.array(embeddings),
        )
        medoids = [text_collection[idx] for idx in medoid_indices]
        metrics.medoid_selection_time_ms = (time.time() -
                                            t_initial_medoid) * 1000
        if self.verbose:
            print(f"Medoid indices: {medoid_indices}")

        # Retry loop for generating a good question
        # The loop attempts to generate a question that leads to an acceptable split ratio
        # If min_split_ratio or max_split_ratio is None, no validation occurs
        if initial_blacklist is not None:
            blacklist = initial_blacklist
        else:
            blacklist = []

        hypothesis = None
        propagated_labels = None
        left_docs = []
        right_docs = []

        doc_indices = []
        for attempt in range(self.max_retries + 1):
            if self.verbose:
                print(
                    f"Attempt {attempt + 1} to generate hypothesis and split documents..."
                )
            # Step 3: Generate hypothesis using LLM
            llm_start = time.time()
            try:
                self.llm_model_config.blacklist = blacklist
                question_asker = makequestion.QuestionMaker(
                    config=self.llm_model_config)
                response = question_asker(collection=medoids)
            except:
                print("LLM call failed during hypothesis generation.")
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

            # Step 4: Use NLI to answer the question for selected documents
            t_initial_kmeans = time.time()
            nli_sampler = sampler_factory(self.nli_sampler_config)
            doc_indices = nli_sampler(
                faiss_index=faiss_index,
                X=np.array(embeddings),
            )

            metrics.kmeans_time_ms += (time.time() - t_initial_kmeans) * 1000

            # Initialize labels as unlabeled (-1)
            answers = -np.ones((len(text_collection), ), dtype=object)

            nli_confidences = []
            nli_start = time.time()
            # Batched NLI
            if self.verbose:
                print("Using batched NLI calls...")

            premises = [
                text_collection[doc_index] for doc_index in doc_indices
            ]
            batched_results = self.nli_batching_model(
                premise=premises,
                hypothesis=hypothesis,
            )
            if self.verbose:
                print(f"Text premises: {len(premises)}")
                print(f"Batched NLI returned {len(batched_results)} results.")

            for i, doc_index in enumerate(doc_indices):
                entails, entailment_score, contradiction_score, P_entailment = batched_results[
                    i]
                answers[doc_index] = 1 if entails else 0
                nli_confidences.append(P_entailment)
                metrics.nli_calls += 1

            metrics.nli_time_ms += (time.time() - nli_start) * 1000

            if self.verbose:
                print(f"NLI answered {len(doc_indices)} documents.")
                print(
                    f"Number of 1s: {np.sum(answers == 1)}, Number of 0s: {np.sum(answers == 0)}"
                )

            # Calculate average medoid NLI confidence
            # Note: This is overwritten on each retry to reflect the final hypothesis's confidence
            if nli_confidences:
                metrics.medoid_nli_confidence_avg = sum(nli_confidences) / len(
                    nli_confidences)

            # Step 5: Propagate labels
            label_prop_start = time.time()
            label_propagation = LabelPropagation(
                faiss_index=faiss_index,
                config=self.label_propagation_config,)
            X = np.array(embeddings).astype('float32')
            y = answers
            propagated_labels = label_propagation.fit_predict(X, y)
            metrics.label_propagation_time_ms += (time.time() -
                                                  label_prop_start) * 1000
            if self.verbose:
                n_labeled = np.sum(answers != -1)
                n_propagated_1 = np.sum(propagated_labels == 1)
                n_propagated_0 = np.sum(propagated_labels == 0)
                print(
                    f"Label propagation completed. Initially labeled: {n_labeled}, Propagated 1s: {n_propagated_1}, Propagated 0s: {n_propagated_0}"
                )

            # Step 6: Build TreeNode based on propagated labels
            # Documents where label == 1 go to left child, label == 0 go to right child
            left_docs = [
                i for i, label in enumerate(propagated_labels) if label == 1
            ]
            right_docs = [
                i for i, label in enumerate(propagated_labels) if label == 0
            ]

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

                # Store split ratio and split entropy (assuming bernoulli prior)
                # for the last attempt
                if attempt == self.max_retries or split_is_valid:
                    metrics.split_ratio = len(left_docs) / len(text_collection)
                    entropy = -(split_ratio * np.log2(split_ratio) +
                                (1 - split_ratio) * np.log2(1 - split_ratio))
                    metrics.split_entropy = entropy

            # If split is valid or we've exhausted retries, stop
            if split_is_valid or attempt == self.max_retries:
                if self.verbose:
                    print("Accepting hypothesis and split.")
                    print(
                        f"Split ratio: {len(left_docs) / len(text_collection):.3f}"
                    )
                    print(
                        f"Left documents: {len(left_docs)}, Right documents: {len(right_docs)}"
                    )
                    print(f"Retries: {attempt}")
                break

            # Otherwise, add this hypothesis to the blacklist and retry
            blacklist.append(hypothesis)

            if self.verbose:
                print(f"Hypothesis '{hypothesis}' led to invalid split.")
                print(
                    f"Split ratio {split_ratio:.3f} not acceptable. Retrying..."
                )

        if self.verbose:
            print(f"Final hypothesis: {hypothesis}")
            print(
                f"Left documents: {len(left_docs)}, Right documents: {len(right_docs)}"
            )
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
            print(
                f"RTPBuilder execution completed in {metrics.total_time_ms:.2f} ms"
            )
            print(f"Total LLM input tokens: {metrics.llm_input_tokens}")
            print(f"Total LLM output tokens: {metrics.llm_output_tokens}")
            print(f"Total NLI calls: {metrics.nli_calls}")
            print(
                f"Total FAISS indexing time: {metrics.faiss_search_time_ms:.2f} ms"
            )
            print(
                f"Total Medoid selection time for LLM: {metrics.medoid_selection_time_ms:.2f} ms"
            )
            print(
                f"Total KMeans time for NLI: {metrics.kmeans_time_ms:.2f} ms")
            print(
                f"Total LLM request time: {metrics.llm_request_time_ms:.2f} ms"
            )
            print(f"Total NLI time: {metrics.nli_time_ms:.2f} ms")
            print(
                f"Total Label Propagation time: {metrics.label_propagation_time_ms:.2f} ms"
            )
            print(f"Total attempts: {metrics.n_attempts}")
            print(
                f"Medoid NLI confidence avg: {metrics.medoid_nli_confidence_avg:.4f}"
            )
            print(f"Split ratio: {metrics.split_ratio:.4f}")
            print(f"Split entropy: {metrics.split_entropy:.4f}")

        if return_metrics:
            return root, metrics
        return root


class RTPRecursion:
    """
    RTPRecursion class that recursively builds RTP trees with stopping criteria.
    
    This class wraps an RTPBuilder and recursively applies it to build a complete
    tree structure with configurable stopping criteria.
    
    Attributes:
        builder: Pre-initialized RTPBuilder instance
        min_node_size: Minimum number of documents for a node to be split
        min_split_ratio: Minimum split ratio (proportion in smaller child)
        max_split_ratio: Maximum split ratio (proportion in smaller child)
        max_depth: Maximum depth of the tree
    """

    def __init__(
        self,
        builder: RTPBuilder,
        min_node_size: int = 2,
        min_split_ratio: float = 0.2,
        max_split_ratio: float = 0.8,
        max_depth: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize RTPRecursion with a builder and stopping criteria.
        
        Args:
            builder: Pre-initialized RTPBuilder instance
            min_node_size: Minimum number of documents required to split a node
            min_split_ratio: Minimum split ratio for a valid split (default: 0.2)
            max_split_ratio: Maximum split ratio for a valid split (default: 0.8)
            max_depth: Maximum depth of the tree (default: 10)
        """
        self.builder = builder
        self.min_node_size = min_node_size
        self.min_split_ratio = min_split_ratio
        self.max_split_ratio = max_split_ratio
        self.max_depth = max_depth
        self.verbose = verbose

    def __call__(self, X: Iterable[str]) -> tuple[TreeNode, SplitMetrics]:
        """
        Recursively build an RTP tree from a collection of text documents.
        
        Args:
            X: Iterable of text documents (strings)
            
        Returns:
            tuple[TreeNode, SplitMetrics]: Root node of the tree and global metrics
        """
        if self.verbose:
            print("Starting RTPRecursion...")

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
            
        Returns:
            tuple[TreeNode, SplitMetrics]: The node for this subset and accumulated metrics
        """
        print("Entering recursion at depth", depth, "with",
              len(document_indices), "documents.")
        # Get the subset of documents for this node
        node_documents = [text_collection[i] for i in document_indices]

        # Check stopping criteria
        should_stop = (len(document_indices) < self.min_node_size
                       or depth >= self.max_depth)

        if should_stop:
            # Create leaf node - leaf nodes don't execute RTPBuilder, so no metrics
            return TreeNode(documents=document_indices), SplitMetrics()

        # Execute RTPBuilder for current node
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
        left_original_indices = [
            document_indices[i] for i in node_root.left.documents
        ]
        right_original_indices = [
            document_indices[i] for i in node_root.right.documents
        ]

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

        # Combine metrics from current node and children using __add__
        combined_metrics = node_metrics + left_metrics + right_metrics

        return node_root, combined_metrics
