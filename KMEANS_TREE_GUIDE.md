# K-Means Tree Building

This document describes the K-means-based tree building functionality in AskMe.

## Overview

The K-means tree building provides an alternative approach to building hierarchical document organization trees using K-means clustering instead of direct hypothesis generation.

## How It Works

The K-means tree building algorithm works as follows:

1. **Embedding**: Gets embeddings for all documents using a sentence transformer model
2. **Clustering**: Runs k-means with k=2 to split documents into 2 clusters
3. **Medoid Selection**: Selects representative documents (medoids) from each cluster
4. **Hypothesis Generation**: Asks an LLM to generate a hypothesis that is true for elements in "cluster 1" but false for elements in "cluster 0"
5. **Validation**: Uses Natural Language Inference (NLI) to validate the hypothesis against documents
6. **Propagation**: Uses label propagation to assign all documents to clusters
7. **Recursion**: Recursively applies the algorithm to each cluster to build a tree

## Classes

### KMeansTreeBuilder

The main class that performs single-level K-means splitting.

**Key Parameters:**
- `use_gpu`: Whether to use GPU acceleration (default: False)
- `embedding_model_name`: Name of the sentence transformer model
- `nli_model_name`: Name of the NLI model
- `llm_model_name`: Name of the LLM model for hypothesis generation
- `n_medoids_per_cluster`: Number of medoids to select from each cluster (default: 2)
- `n_documents_to_answer`: Number of documents to label with NLI
- `knn_neighbors`: Number of neighbors for k-NN graph
- `alpha`: Alpha parameter for label propagation (0 < alpha < 1)
- `max_retries`: Maximum number of retries if split ratio is bad
- `min_split_ratio`: Minimum acceptable split ratio (proportion in smaller child)
- `max_split_ratio`: Maximum acceptable split ratio (proportion in smaller child)
- `verbose`: Whether to print verbose output during execution

**Example:**
```python
from askme.rtp import KMeansTreeBuilder

# Initialize the builder
builder = KMeansTreeBuilder(
    use_gpu=False,
    n_medoids_per_cluster=2,
    n_documents_to_answer=6,
)

# Build a tree
text_collection = ["doc1", "doc2", "doc3", ...]
tree_root = builder(text_collection)

# With metrics
tree_root, metrics = builder(text_collection, return_metrics=True)
```

### KMeansTreeRecursion

A wrapper class that recursively applies KMeansTreeBuilder to build a complete tree structure with configurable stopping criteria.

**Key Parameters:**
- `builder`: Pre-initialized KMeansTreeBuilder instance
- `min_node_size`: Minimum number of documents required to split a node
- `min_split_ratio`: Minimum split ratio for a valid split (default: 0.2)
- `max_split_ratio`: Maximum split ratio for a valid split (default: 0.8)
- `max_depth`: Maximum depth of the tree (default: 10)
- `verbose`: Whether to print verbose output

**Example:**
```python
from askme.rtp import KMeansTreeBuilder, KMeansTreeRecursion

# Initialize the builder
builder = KMeansTreeBuilder(use_gpu=False)

# Initialize recursion with stopping criteria
recursion = KMeansTreeRecursion(
    builder=builder,
    min_node_size=3,
    min_split_ratio=0.2,
    max_split_ratio=0.8,
    max_depth=5,
)

# Build a recursive tree
tree_root, global_metrics = recursion(text_collection)
```

## Differences from RTPBuilder

The main difference between K-means tree building and RTP (Retrieval-based Tree Partitioning) is in how documents are initially grouped:

- **RTPBuilder**: Selects medoids from the entire collection and generates a hypothesis directly
- **KMeansTreeBuilder**: First clusters documents using k-means (k=2), then generates a hypothesis that distinguishes between the two clusters

This approach can lead to more balanced splits and better cluster separation in some cases.

## Medoid Selection Strategies

Both builders support multiple strategies for selecting representative documents:

- `'kmeans'`: Use k-means clustering to find representative documents (default)
- `'random'`: Randomly select documents
- `'votek'`: Use vote-k sampling to find diverse representatives

Set via the `selection_strategy` parameter.

## Label Propagation

After NLI labels a subset of documents, label propagation is used to assign labels to all documents based on:
- A k-NN graph of document embeddings
- Semi-supervised learning with the `alpha` parameter controlling the balance between labeled data and graph structure

## Metrics

Both classes return `SplitMetrics` objects when `return_metrics=True`, containing:
- `llm_input_tokens`: Number of input tokens sent to LLM
- `llm_output_tokens`: Number of output tokens from LLM
- `nli_calls`: Number of NLI inference calls
- `faiss_search_time_ms`: Time spent on FAISS operations
- `kmeans_time_ms`: Time spent on k-means clustering
- `medoid_selection_time_ms`: Time spent selecting medoids
- `llm_request_time_ms`: Time spent on LLM requests
- `nli_time_ms`: Time spent on NLI inference
- `label_propagation_time_ms`: Time spent on label propagation
- `total_time_ms`: Total execution time
- `split_ratio`: Proportion of documents in left vs right child
- `split_entropy`: Entropy of the split
- `medoid_nli_confidence_avg`: Average NLI confidence for medoids
- `n_attempts`: Number of attempts to generate a valid split

For recursive building, metrics are accumulated across all nodes.

## Examples

See the following example files:
- `examples/use_kmeans_tree_builder.py`: Basic usage of KMeansTreeBuilder
- `examples/use_kmeans_tree_recursion.py`: Recursive tree building with KMeansTreeRecursion

## Testing

Run the test suite:
```bash
pytest tests/test_kmeans_tree_builder.py
```

Note: Tests are marked with `@pytest.mark.llm` as they require LLM API access.
