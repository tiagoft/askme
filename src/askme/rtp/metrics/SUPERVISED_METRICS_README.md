# Supervised Metrics for RTP Trees

This module provides supervised (label-aware) metrics for evaluating RTP tree quality when ground truth labels are available.

## Overview

Supervised metrics use ground truth labels to measure how well a tree structure captures the underlying class structure of the data. These metrics are useful for:

- **Comparing tree building strategies** - Determine which approach produces better class separation
- **Evaluating tree quality** - Quantify how well the tree aligns with ground truth
- **Understanding class separation** - Identify which classes are well-separated in the tree

## Available Metrics

### Clustering Metrics

These metrics treat each tree node (leaf or internal) as a cluster and measure agreement with true labels:

1. **Normalized Mutual Information (NMI)**
   - Measures mutual information between cluster assignments and true labels
   - Range: [0, 1], higher is better
   - 1.0 = perfect agreement, 0.0 = no mutual information
   - Popular in clustering evaluation (HERCULES, many 2024-2025 papers)

2. **Adjusted Rand Index (ARI)**
   - Measures similarity between clusterings, adjusted for chance
   - Range: [-1, 1], higher is better
   - 1.0 = perfect agreement, 0.0 = random labeling, negative = worse than random
   - Very popular in recent papers (HERCULES, many 2024-2025 papers)

3. **Homogeneity, Completeness, V-measure**
   - **Homogeneity**: Each cluster contains only members of a single class
   - **Completeness**: All members of a class are in the same cluster
   - **V-measure**: Harmonic mean of homogeneity and completeness
   - Range: [0, 1] for each metric, higher is better

### Classification Metrics

These metrics treat the tree as a classifier where each document is assigned the majority class of its node:

4. **Accuracy (ACC)**
   - Proportion of correctly classified documents
   - Range: [0, 1], higher is better

5. **F1-Score**
   - Harmonic mean of precision and recall
   - Supports multiple averaging methods: 'micro', 'macro', 'weighted'
   - Range: [0, 1], higher is better

6. **Confusion Matrix**
   - Shows true vs predicted class counts
   - Entry [i, j] = number of true class i predicted as class j
   - Useful for understanding which classes are confused

## Usage

### Basic Usage

```python
from askme.rtp.tree_models import TreeNode
from askme.rtp.supervised_metrics import (
    NormalizedMutualInformation,
    AdjustedRandIndex,
    Accuracy,
    F1Score,
)

# Assume you have a tree and ground truth labels
tree_root = TreeNode(documents=[0, 1, 2, 3])
labels = [0, 0, 1, 1]  # Ground truth labels for documents 0-3

# Calculate NMI (using leaf nodes only)
nmi_metric = NormalizedMutualInformation()
nmi = nmi_metric.call(tree_root, labels, use_leaves_only=True)
print(f"NMI: {nmi:.4f}")

# Calculate ARI (using leaf nodes only)
ari_metric = AdjustedRandIndex()
ari = ari_metric.call(tree_root, labels, use_leaves_only=True)
print(f"ARI: {ari:.4f}")

# Calculate Accuracy
acc_metric = Accuracy()
acc = acc_metric.call(tree_root, labels, use_leaves_only=True)
print(f"Accuracy: {acc:.4f}")

# Calculate F1-Score with weighted averaging
f1_metric = F1Score()
f1 = f1_metric.call(tree_root, labels, use_leaves_only=True, average='weighted')
print(f"F1-Score: {f1:.4f}")
```

### Using All Nodes vs Leaf Nodes Only

By default, metrics use only leaf nodes (`use_leaves_only=True`). This is recommended for most use cases as leaf nodes represent the final clustering/classification.

You can also evaluate using all nodes (internal + leaf):

```python
# Using leaf nodes only (recommended)
nmi_leaves = nmi_metric.call(tree_root, labels, use_leaves_only=True)

# Using all nodes (includes internal nodes)
nmi_all = nmi_metric.call(tree_root, labels, use_leaves_only=False)
```

**Note:** When using all nodes, documents may appear in multiple clusters (their leaf node and all ancestor nodes), which can make interpretation more complex.

### Homogeneity, Completeness, and V-measure

```python
from askme.rtp.supervised_metrics import HomogeneityCompletenessVMeasure

hcv_metric = HomogeneityCompletenessVMeasure()
hcv = hcv_metric.call(tree_root, labels, use_leaves_only=True)

print(f"Homogeneity: {hcv['homogeneity']:.4f}")
print(f"Completeness: {hcv['completeness']:.4f}")
print(f"V-measure: {hcv['v_measure']:.4f}")
```

### Confusion Matrix

```python
from askme.rtp.supervised_metrics import ConfusionMatrix

cm_metric = ConfusionMatrix()
cm = cm_metric.call(tree_root, labels, use_leaves_only=True)

print("Confusion Matrix:")
print(cm)
# Output: numpy array where entry [i,j] = count of true class i predicted as j
```

### NMI with Different Averaging Methods

```python
# Arithmetic mean (default)
nmi_arith = nmi_metric.call(tree_root, labels, average_method='arithmetic')

# Geometric mean
nmi_geom = nmi_metric.call(tree_root, labels, average_method='geometric')

# Min/Max
nmi_min = nmi_metric.call(tree_root, labels, average_method='min')
nmi_max = nmi_metric.call(tree_root, labels, average_method='max')
```

### F1-Score with Different Averaging Methods

```python
# Weighted average (default) - accounts for class imbalance
f1_weighted = f1_metric.call(tree_root, labels, average='weighted')

# Macro average - treats all classes equally
f1_macro = f1_metric.call(tree_root, labels, average='macro')

# Micro average - aggregates contributions of all classes
f1_micro = f1_metric.call(tree_root, labels, average='micro')
```

## Architecture

The module follows the specified interface:

1. **Base Class**: `SupervisedMetric`
   - Abstract base class with `call(self, root, labels, **kwargs)` method
   - All metrics inherit from this class

2. **Implementation Pattern**:
   ```python
   def _metric_implementation(root, labels, **kwargs):
       # Actual implementation using sklearn
       pass
   
   class MetricName(SupervisedMetric):
       def call(self, root, labels, **kwargs):
           return _metric_implementation(root, labels, **kwargs)
   ```

3. **Helper Functions**:
   - `_get_cluster_assignments()` - Maps documents to cluster IDs (for clustering metrics)
   - `_get_predicted_labels()` - Maps documents to predicted labels via majority voting (for classification metrics)
   - `get_all_nodes()` - Retrieves all nodes in the tree
   - `get_all_leaves()` - Retrieves only leaf nodes

## Example

See `examples/demo_supervised_metrics.py` for a complete working example that demonstrates all metrics.

Run it with:
```bash
python examples/demo_supervised_metrics.py
```

## Testing

Comprehensive tests are available in `tests/test_supervised_metrics.py`. Run them with:

```bash
pytest tests/test_supervised_metrics.py -v
```

The test suite includes:
- Tests for each metric with perfect and imperfect clustering/classification
- Edge cases (empty trees, single nodes)
- Multi-class problems
- Different parameter configurations
- Comparison of leaf-only vs all-nodes evaluation

## Implementation Details

### How Clustering Metrics Work

For clustering metrics (NMI, ARI, H/C/V), each node in the tree is treated as a cluster:
1. Extract all relevant nodes (leaves or all nodes based on `use_leaves_only`)
2. Assign each document a cluster ID based on which node contains it
3. Use sklearn's clustering metrics to compare cluster assignments with true labels

### How Classification Metrics Work

For classification metrics (Accuracy, F1, Confusion Matrix):
1. Extract all relevant nodes (leaves or all nodes based on `use_leaves_only`)
2. For each node, find the majority class among its documents
3. Assign each document the majority class of its node (predicted label)
4. Use sklearn's classification metrics to compare predictions with true labels

### sklearn Implementation

All metrics use scikit-learn implementations where possible:
- `sklearn.metrics.normalized_mutual_info_score` for NMI
- `sklearn.metrics.adjusted_rand_score` for ARI
- `sklearn.metrics.homogeneity_completeness_v_measure` for H/C/V
- `sklearn.metrics.accuracy_score` for Accuracy
- `sklearn.metrics.f1_score` for F1-Score
- `sklearn.metrics.confusion_matrix` for Confusion Matrix

This ensures:
- Well-tested, standard implementations
- Consistent behavior with other ML libraries
- Good performance on large datasets

## References

- Normalized Mutual Information: Used extensively in clustering evaluation literature
- Adjusted Rand Index: Very popular in recent papers including HERCULES and many 2024-2025 publications
- Homogeneity, Completeness, V-measure: Rosenberg and Hirschberg (2007)
- Standard classification metrics: Confusion matrix, Accuracy, F1-Score

## See Also

- `evaluator.py` - Unsupervised (self-supervised) metrics like purity and entropy
- `tree_models.py` - TreeNode data structure definition
- `examples/demo_evaluator.py` - Example of unsupervised metrics
