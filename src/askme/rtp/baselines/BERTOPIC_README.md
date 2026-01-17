# BERTopic Baseline with Hierarchical Clustering

This module provides a baseline clustering approach using BERTopic with hierarchical clustering for comparison with RTP and HDBSCAN methods.

## Overview

The BERTopic baseline:
1. Vectorizes text documents using a sentence transformer model
2. Clusters the embeddings using BERTopic with topic modeling
3. Creates a hierarchical structure based on topic relationships
4. Converts the clustering results into a binary tree structure (TreeNode)
5. Enables evaluation using the same metrics as RTP trees (purity, entropy, isolation depth)

## What is BERTopic?

BERTopic is a topic modeling technique that leverages transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.

Key features:
- Uses sentence transformers for document embeddings
- Applies clustering (HDBSCAN or hierarchical clustering)
- Generates topic representations using c-TF-IDF
- Can create hierarchical topic structures for multi-level organization
- Identifies outliers that don't fit into any topic

## Usage

### Basic Usage

```python
from askme.rtp import run_bertopic_baseline, evaluate_exploratory_power, calculate_tree_depth

# Your text documents
texts = [
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "Machine learning is fascinating.",
    "Neural networks can learn patterns.",
]

# Run BERTopic baseline
tree, embeddings, topic_model = run_bertopic_baseline(
    texts,
    model_name="all-MiniLM-L6-v2",  # Sentence transformer model
    nr_topics="auto",                # Number of topics (or "auto")
    device="cpu",                    # or "cuda" for GPU
)

# Calculate tree depth
depth = calculate_tree_depth(tree)
print(f"Tree depth: {depth}")

# Evaluate the tree (requires labels)
labels = [0, 0, 1, 1]  # Your document labels
results = evaluate_exploratory_power(tree, labels)
print(f"Average leaf purity: {results['average_leaf_purity']:.4f}")
print(f"Average leaf entropy: {results['average_leaf_entropy']:.4f}")

# Access topic information
topic_info = topic_model.get_topic_info()
print(topic_info)
```

### Running the Demo

```bash
python examples/demo_bertopic_baseline.py
```

This demo script:
- Creates a sample dataset with three categories (Animals, Technology, Sports)
- Runs BERTopic clustering with hierarchical structure
- Evaluates the resulting tree structure
- Displays purity, entropy, and isolation depth metrics
- Shows topic information discovered by BERTopic

## Functions

### `run_bertopic_baseline`

Main function that runs the complete BERTopic baseline pipeline.

**Parameters:**
- `texts` (List[str]): List of text strings to cluster
- `model_name` (str): Name of the sentence transformer model (default: "all-MiniLM-L6-v2")
- `nr_topics` (str or int): Number of topics to create, "auto" for automatic (default: "auto")
- `device` (str): Device to run the model on ('cpu' or 'cuda', default: 'cpu')
- `calculate_probabilities` (bool): Whether to calculate topic probabilities (default: False)

**Returns:**
- Tuple of (TreeNode, embeddings array, BERTopic model)

### `vectorize_texts`

Vectorizes text documents using a sentence transformer model.

**Parameters:**
- `texts` (List[str]): List of text strings to vectorize
- `model_name` (str): Name of the sentence transformer model
- `device` (str): Device to run the model on

**Returns:**
- NumPy array of embeddings with shape (n_texts, embedding_dim)

### `build_tree_from_bertopic_hierarchy`

Converts BERTopic clustering results into a binary TreeNode structure.

**Parameters:**
- `topic_model`: Fitted BERTopic model
- `topics` (List[int]): Topic assignments for each document
- `n_samples` (int): Number of samples in the dataset

**Returns:**
- Root TreeNode of the constructed tree

### `calculate_tree_depth`

Calculates the maximum depth of a tree.

**Parameters:**
- `root` (TreeNode): Root node of the tree

**Returns:**
- Maximum depth of the tree (root has depth 0)

## Evaluation Metrics

The BERTopic baseline can be evaluated using the same metrics as RTP trees:

- **Purity**: Measures how well each leaf node contains documents of a single class (1.0 = perfect, 0.0 = completely mixed)
- **Entropy**: Measures the disorder/mixing of classes within each leaf node (0.0 = perfect, higher = more mixed)
- **Isolation Depth**: Shows how deep in the tree each class becomes separated
- **Tree Depth**: The maximum depth of the hierarchical structure

## Comparison with RTP and HDBSCAN

The BERTopic baseline provides a topic modeling approach that:

**Advantages:**
- Generates interpretable topics with word representations
- Can automatically determine the number of topics
- Creates hierarchical topic relationships
- Identifies outliers/noise documents
- Topics provide semantic descriptions of clusters

**Disadvantages:**
- Less semantic than RTP's LLM-generated questions
- More complex than simple distance-based clustering (HDBSCAN)
- Requires tuning of topic modeling parameters
- No LLM reasoning about document relationships

**Comparison Summary:**
- **RTP**: Semantic questions for splits (most interpretable but expensive)
- **HDBSCAN**: Pure density-based clustering (fast, no semantics)
- **BERTopic**: Topic-based clustering (interpretable topics, moderate cost)

## Dependencies

- bertopic >= 0.16.0
- scikit-learn >= 1.3.0
- sentence-transformers >= 5.2.0
- numpy

## Testing

Run tests with:

```bash
python tests/test_bertopic_baseline_lightweight.py
```

The lightweight tests use mocks to validate core logic without heavy dependencies.

## Example Comparison

To compare RTP, HDBSCAN, and BERTopic on the AG News dataset:

```bash
python experiments/rtp_evaluation/test_small_agnews.py
```

This script will:
1. Load 100 samples from AG News
2. Run all three methods (RTP, HDBSCAN, BERTopic)
3. Evaluate each with purity, entropy, and isolation depth
4. Display a comprehensive comparison

## References

- BERTopic: Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
- BERTopic Documentation: https://maartengr.github.io/BERTopic/
- Sentence Transformers: https://www.sbert.net/
