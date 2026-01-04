# HDBSCAN Baseline

This module provides a baseline clustering approach using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) for comparison with RTP methods.

## Overview

The HDBSCAN baseline:
1. Vectorizes text documents using a sentence transformer model
2. Clusters the embeddings using HDBSCAN
3. Converts the clustering results into a binary tree structure (TreeNode)
4. Enables evaluation using the same metrics as RTP trees (purity, entropy, isolation depth)

## Usage

### Basic Usage

```python
from askme.rtp import run_hdbscan_baseline, evaluate_exploratory_power, calculate_tree_depth

# Your text documents
texts = [
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "Machine learning is fascinating.",
    "Neural networks can learn patterns.",
]

# Run HDBSCAN baseline
tree, embeddings = run_hdbscan_baseline(
    texts,
    model_name="all-MiniLM-L6-v2",  # Sentence transformer model
    min_cluster_size=2,              # Minimum cluster size
    min_samples=1,                   # Minimum samples for core points
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
```

### Running the Demo

```bash
python examples/demo_hdbscan_baseline.py
```

This demo script:
- Creates a sample dataset with three categories (Animals, Technology, Sports)
- Runs HDBSCAN clustering
- Evaluates the resulting tree structure
- Displays purity, entropy, and isolation depth metrics

## Functions

### `run_hdbscan_baseline`

Main function that runs the complete HDBSCAN baseline pipeline.

**Parameters:**
- `texts` (List[str]): List of text strings to cluster
- `model_name` (str): Name of the sentence transformer model (default: "all-MiniLM-L6-v2")
- `min_cluster_size` (int): Minimum cluster size for HDBSCAN (default: 5)
- `min_samples` (int): Minimum samples parameter for HDBSCAN (default: 1)
- `device` (str): Device to run the model on ('cpu' or 'cuda', default: 'cpu')

**Returns:**
- Tuple of (TreeNode, embeddings array)

### `vectorize_texts`

Vectorizes text documents using a sentence transformer model.

**Parameters:**
- `texts` (List[str]): List of text strings to vectorize
- `model_name` (str): Name of the sentence transformer model
- `device` (str): Device to run the model on

**Returns:**
- NumPy array of embeddings with shape (n_texts, embedding_dim)

### `build_tree_from_hdbscan`

Converts HDBSCAN clustering results into a binary TreeNode structure.

**Parameters:**
- `clusterer`: Fitted HDBSCAN clusterer
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

The HDBSCAN baseline can be evaluated using the same metrics as RTP trees:

- **Purity**: Measures how well each leaf node contains documents of a single class (1.0 = perfect, 0.0 = completely mixed)
- **Entropy**: Measures the disorder/mixing of classes within each leaf node (0.0 = perfect, higher = more mixed)
- **Isolation Depth**: Shows how deep in the tree each class becomes separated
- **Tree Depth**: The maximum depth of the hierarchical structure

## Comparison with RTP

The HDBSCAN baseline provides a density-based hierarchical clustering approach that:

**Advantages:**
- Automatically determines the number of clusters
- Can identify noise/outliers
- Works well with clusters of varying densities
- No LLM or NLI model required

**Disadvantages:**
- No semantic questions to describe splits
- Less interpretable than RTP's question-based splits
- Depends on distance-based clustering rather than semantic understanding
- May not capture complex categorical distinctions as well as RTP

## Dependencies

- scikit-learn >= 1.3.0
- sentence-transformers >= 5.2.0
- numpy

## Testing

Run tests with:

```bash
pytest tests/test_tree_depth.py -v
```

For full integration tests (requires all dependencies):
```bash
pytest tests/test_hdbscan_baseline.py -v
```

## References

- HDBSCAN Paper: Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates.
- Scikit-learn HDBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
- Sentence Transformers: https://www.sbert.net/
