# Self-Supervised Tree Metrics

This module provides metrics to evaluate RTP tree quality using only document embeddings and tree structure, without requiring ground truth labels.

## Overview

All metrics inherit from the `SelfSupervisedMetric` base class and implement a `call(root, embeddings, **kwargs)` method.

## Available Metrics

### 1. Clustering Metrics

These metrics treat leaf nodes as clusters and evaluate clustering quality:

#### SilhouetteScoreMetric
Measures how similar documents are to their own cluster compared to other clusters.
- **Range**: -1 to 1
- **Interpretation**: Higher is better
- **Best for**: Evaluating overall cluster separation

```python
from askme.rtp import SilhouetteScoreMetric

metric = SilhouetteScoreMetric()
score = metric.call(tree_root, embeddings)
```

#### DaviesBouldinScoreMetric
Measures the average similarity ratio of each cluster with its most similar cluster.
- **Range**: 0 or higher
- **Interpretation**: Lower is better
- **Best for**: Detecting overlapping or poorly separated clusters

```python
from askme.rtp import DaviesBouldinScoreMetric

metric = DaviesBouldinScoreMetric()
score = metric.call(tree_root, embeddings)
```

#### CalinskiHarabaszScoreMetric
Variance ratio: between-cluster dispersion to within-cluster dispersion.
- **Range**: 0 or higher
- **Interpretation**: Higher is better
- **Best for**: Evaluating cluster density and separation

```python
from askme.rtp import CalinskiHarabaszScoreMetric

metric = CalinskiHarabaszScoreMetric()
score = metric.call(tree_root, embeddings)
```

### 2. Topic Diversity Metric

Calculates the proportion of unique words across all top-k words of all topics (questions).

- **Range**: 0 to 1
- **Interpretation**: Higher is better (more diverse, less redundancy)
- **Modes**:
  - `full_tree`: Calculate diversity across all questions in the tree
  - `leaf_paths`: Calculate per-leaf path and average across all leaves

```python
from askme.rtp import TopicDiversityMetric

metric = TopicDiversityMetric()

# Full tree mode
score_full = metric.call(tree_root, embeddings, mode="full_tree", topk=10)

# Leaf paths mode
score_leaf = metric.call(tree_root, embeddings, mode="leaf_paths", topk=10)

# With custom stop words and minimum word length
custom_stop_words = {'the', 'is', 'a', 'an'}
score = metric.call(
    tree_root, 
    embeddings, 
    topk=10,
    min_word_length=3,
    stop_words=custom_stop_words
)
```

### 3. Child-Parent Uniqueness Metric

Measures how distinct child nodes are from their parent nodes using cosine similarity of average embeddings.

- **Returns**: Dictionary with:
  - `avg_cosine_similarity`: 0 to 1 (lower means more unique)
  - `avg_uniqueness`: 0 to 1 (higher means more unique)
  - `num_parent_child_pairs`: Number of pairs evaluated

```python
from askme.rtp import ChildParentUniquenessMetric

metric = ChildParentUniquenessMetric()
result = metric.call(tree_root, embeddings)

print(f"Average uniqueness: {result['avg_uniqueness']:.3f}")
print(f"Pairs evaluated: {result['num_parent_child_pairs']}")
```

## Complete Example

```python
import numpy as np
from askme.rtp import (
    TreeNode,
    SilhouetteScoreMetric,
    DaviesBouldinScoreMetric,
    CalinskiHarabaszScoreMetric,
    TopicDiversityMetric,
    ChildParentUniquenessMetric,
)

# Create a tree structure
root = TreeNode(
    documents=[0, 1, 2, 3, 4, 5],
    question="Is this about technology?"
)
left = TreeNode(documents=[0, 1, 2], question="Is this about AI?")
right = TreeNode(documents=[3, 4, 5], question="Is this about hardware?")
root.left = left
root.right = right

# Create embeddings (6 documents, 128 dimensions)
embeddings = np.random.randn(6, 128).astype('float32')

# Compute all metrics
silhouette = SilhouetteScoreMetric()
davies_bouldin = DaviesBouldinScoreMetric()
calinski = CalinskiHarabaszScoreMetric()
diversity = TopicDiversityMetric()
uniqueness = ChildParentUniquenessMetric()

# Get scores
sil_score = silhouette.call(root, embeddings)
db_score = davies_bouldin.call(root, embeddings)
ch_score = calinski.call(root, embeddings)
div_score = diversity.call(root, embeddings, mode="full_tree")
uniq = uniqueness.call(root, embeddings)

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Score: {db_score:.3f}")
print(f"Calinski-Harabasz Score: {ch_score:.3f}")
print(f"Topic Diversity: {div_score:.3f}")
print(f"Child-Parent Uniqueness: {uniq['avg_uniqueness']:.3f}")
```

## Requirements

- All clustering metrics require at least 2 leaf nodes (clusters)
- All documents must be assigned to leaf nodes
- Topic diversity requires questions/hypotheses to be set on tree nodes
- Embeddings should be numpy arrays of shape (n_documents, embedding_dim)

## Notes

- Clustering metrics use scikit-learn implementations
- Topic diversity uses simple word tokenization (can be customized with stop_words parameter)
- Child-parent uniqueness computes average embeddings per node for comparison
