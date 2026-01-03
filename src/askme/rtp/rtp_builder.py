"""RTPBuilder class for operationalizing the RTP (Retrieval-based Tree Partitioning) algorithm."""

import faiss
import numpy as np
import time
from collections.abc import Iterable
from typing import Optional, Union

from .make_collection_index import make_faiss_index
from .label_propagation import propagate_labels, make_knn_graph, sparse_affinity
from .tree_models import TreeNode, SplitMetrics
from ..utils import TextEmbeddingWithChunker, kmeans_with_faiss
from ..makequestions import api, makequestion
from ..askquestions import check_entailment, models


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
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        embedding_model_name: str = 'sentence-transformers/paraphrase-albert-small-v2',
        nli_model_name: str = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
        llm_model_name: str = "gpt-4o-mini",
        chunk_size: int = 50,
        overlap: int = 10,
        n_medoids: int = 4,
        n_documents_to_answer: int = 6,
        knn_neighbors: int = 2,
        alpha: float = 0.99,
        max_iter: int = 100,
        tol: float = 1e-3,
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
            alpha: Alpha parameter for label propagation (0 < alpha < 1)
            max_iter: Maximum iterations for label propagation
            tol: Tolerance for label propagation convergence
        """
        self.use_gpu = use_gpu
        self.embedding_model_name = embedding_model_name
        self.nli_model_name = nli_model_name
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n_medoids = n_medoids
        self.n_documents_to_answer = n_documents_to_answer
        self.knn_neighbors = knn_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
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
        
        # Initialize NLI model
        self.nli_model, self.nli_tokenizer = models.make_nli_model(
            model_name=nli_model_name
        )
        if use_gpu:
            self.nli_model = self.nli_model.to('cuda')
        
        # Initialize LLM model
        self.llm_model = api.make_model(llm_model_name)
    
    def __call__(
        self, X: Iterable[str], return_metrics: bool = False
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
        dimension = self.embedding_model.model.get_sentence_embedding_dimension()
        faiss_index, embeddings = make_faiss_index(
            text_collection=text_collection,
            embedding_model=self.embedding_model,
            dimension=dimension,
            use_gpu=self.use_gpu,
            return_embeddings=True,
            gpu_resources=self.gpu_resources,
        )
        metrics.faiss_search_time_ms = (time.time() - faiss_start) * 1000
        
        # Step 2: Get medoids via k-means for hypothesis generation
        n_clusters = min(self.n_medoids, len(text_collection))
        medoid_indices = kmeans_with_faiss(
            faiss_index=faiss_index,
            X=embeddings,
            n_clusters=n_clusters,
        )
        
        medoids = [text_collection[idx] for idx in medoid_indices]
        
        # Step 3: Generate hypothesis using LLM
        llm_start = time.time()
        response = makequestion.make_a_question_about_collection(
            collection=medoids,
            model=self.llm_model,
            retries=5,
        )
        hypothesis = response.output.hypothesis
        metrics.llm_request_time = (time.time() - llm_start) * 1000
        
        # Track LLM tokens
        metrics.llm_input_tokens = response.usage().input_tokens
        metrics.llm_output_tokens = response.usage().output_tokens
        
        # Step 4: Use NLI to answer the question for selected documents
        n_docs_to_label = min(self.n_documents_to_answer, len(text_collection))
        doc_indices = kmeans_with_faiss(
            faiss_index=faiss_index,
            X=embeddings,
            n_clusters=n_docs_to_label,
        )
        
        # Initialize labels as unlabeled (-1)
        answers = -np.ones((len(text_collection),), dtype=object)
        
        device = 'cuda' if self.use_gpu else 'cpu'
        nli_confidences = []
        nli_start = time.time()
        for doc_index in doc_indices:
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
        metrics.nli_time = (time.time() - nli_start) * 1000
        
        # Calculate average medoid NLI confidence
        if nli_confidences:
            metrics.medoid_nli_confidence_avg = sum(nli_confidences) / len(nli_confidences)
        
        # Step 5: Propagate labels
        label_prop_start = time.time()
        indices, distances = make_knn_graph(
            np.array(embeddings).astype('float32'),
            faiss_index,
            n_neighbors=self.knn_neighbors
        )
        W = sparse_affinity(indices, distances, sigma=1.0)
        propagated_labels = propagate_labels(
            W, answers,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol
        )
        metrics.label_propagation_time_ms = (time.time() - label_prop_start) * 1000
        
        # Step 6: Build TreeNode based on propagated labels
        # Documents where label == 1 go to left child, label == 0 go to right child
        left_docs = [i for i, label in enumerate(propagated_labels) if label == 1]
        right_docs = [i for i, label in enumerate(propagated_labels) if label == 0]
        
        # Calculate split ratio (proportion in left child)
        # If no split occurs (all docs in one child), ratio remains 0.0
        if len(left_docs) > 0 and len(right_docs) > 0:
            metrics.split_ratio = len(left_docs) / len(text_collection)
        
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
    
    def __call__(
        self, X: Iterable[str]
    ) -> tuple[TreeNode, SplitMetrics]:
        """
        Recursively build an RTP tree from a collection of text documents.
        
        Args:
            X: Iterable of text documents (strings)
            
        Returns:
            tuple[TreeNode, SplitMetrics]: Root node of the tree and global metrics
        """
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
        # Get the subset of documents for this node
        node_documents = [text_collection[i] for i in document_indices]
        
        # Check stopping criteria
        should_stop = (
            len(document_indices) < self.min_node_size or
            depth >= self.max_depth
        )
        
        if should_stop:
            # Create leaf node - leaf nodes don't execute RTPBuilder, so no metrics
            return TreeNode(documents=document_indices), SplitMetrics()
        
        # Execute RTPBuilder for current node
        node_root, node_metrics = self.builder(node_documents, return_metrics=True)
        
        # Create tree node with original document indices
        result_node = TreeNode(
            documents=document_indices,
            question=node_root.question,
            metrics=node_metrics,
        )
        
        # Check if split is valid
        if node_root.left is None or node_root.right is None:
            # No split occurred
            return result_node, node_metrics
        
        # Check split ratio criteria
        left_count = len(node_root.left.documents)
        right_count = len(node_root.right.documents)
        total_count = left_count + right_count
        
        if total_count == 0:
            return result_node, node_metrics
        
        # Calculate split ratio (proportion in smaller child)
        smaller_count = min(left_count, right_count)
        split_ratio = smaller_count / total_count
        
        # Check if split ratio is within acceptable range
        if split_ratio < self.min_split_ratio or split_ratio > self.max_split_ratio:
            # Split ratio not acceptable, don't recurse
            return result_node, node_metrics
        
        # Map local indices back to original indices
        left_original_indices = [document_indices[i] for i in node_root.left.documents]
        right_original_indices = [document_indices[i] for i in node_root.right.documents]
        
        # Recurse into children and accumulate metrics
        result_node.left, left_metrics = self._recurse(
            text_collection=text_collection,
            document_indices=left_original_indices,
            depth=depth + 1,
        )
        
        result_node.right, right_metrics = self._recurse(
            text_collection=text_collection,
            document_indices=right_original_indices,
            depth=depth + 1,
        )
        
        # Combine metrics from current node and children using __add__
        combined_metrics = node_metrics + left_metrics + right_metrics
        
        return result_node, combined_metrics
