# Exploratory Power Evaluator

This module provides functions to evaluate the "Exploratory Power" of RTP (Retrieval-based Tree Partitioning) trees. The evaluator calculates metrics that measure how well a tree separates documents based on their labels.

## Metrics

### 1. Node Purity

Node purity measures how homogeneous a leaf node is in terms of class labels. It's calculated as the proportion of the most common class:

```
Purity = proportion of the most common class (p_max)
```

where:
- Impurity = probability that a randomly chosen element does NOT belong to the most common class
- Purity = proportion of the most common class

- **Purity = 1.0**: The node is perfectly pure (all documents belong to the same class)
- **Purity approaches 0**: The node becomes more mixed across classes

### 2. Node Entropy

Entropy is a measure of disorder/impurity in the node:

```
Entropy = -Σ(p_i * log₂(p_i)) for each class i
```

where p_i is the proportion of documents of class i in the node.

- **Entropy = 0**: The node is perfectly pure (all documents belong to the same class)
- **Higher entropy**: The node is more mixed across classes

### 3. Isolation Depth

Isolation depth measures the minimum depth at which a specific class becomes fully isolated from all other classes. A class is "isolated" when a node contains only documents of that class.

- **Lower depth**: The class is separated earlier in the tree (better separation)
- **None**: The class is never fully isolated in any node

## Usage

### Basic Example

```python
from askme.rtp import TreeNode, evaluate_exploratory_power

# Create a tree structure
root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
left_child = TreeNode(documents=[0, 1, 2])
right_child = TreeNode(documents=[3, 4, 5])
root.left = left_child
root.right = right_child

# Define labels for the documents
labels = [0, 0, 0, 1, 1, 1]  # Three docs of class 0, three of class 1

# Evaluate the tree
results = evaluate_exploratory_power(root, labels)

print(f"Average leaf purity: {results['average_leaf_purity']}")
print(f"Average leaf entropy: {results['average_leaf_entropy']}")
print(f"Isolation depths: {results['isolation_depths']}")
```

### Individual Functions

You can also use individual functions for more granular control:

```python
from askme.rtp import (
    calculate_node_purity,
    calculate_node_entropy,
    calculate_all_leaf_purities,
    calculate_all_leaf_entropies,
    calculate_isolation_depth,
    calculate_all_isolation_depths,
)

# Calculate purity for a single node
node = TreeNode(documents=[0, 1, 2])
labels = [0, 0, 1]
purity = calculate_node_purity(node, labels)  # Returns: 0.667 (2/3 are class 0)

# Calculate entropy for a single node
entropy = calculate_node_entropy(node, labels)  # Returns: ~0.918

# Calculate purities for all leaves
leaf_purities = calculate_all_leaf_purities(root, labels)

# Calculate entropies for all leaves
leaf_entropies = calculate_all_leaf_entropies(root, labels)

# Calculate purities for all leaves
leaf_purities = calculate_all_leaf_purities(root, labels)

# Calculate entropies for all leaves
leaf_entropies = calculate_all_leaf_entropies(root, labels)

# Calculate isolation depth for a specific class
depth = calculate_isolation_depth(root, labels, target_class=0)

# Calculate isolation depths for all classes
all_depths = calculate_all_isolation_depths(root, labels)
```

## API Reference

### `evaluate_exploratory_power(root, labels)`

Main evaluation function that calculates node purity, entropy, and isolation depth metrics.

**Parameters:**
- `root` (TreeNode): The root node of the tree
- `labels` (List[int]): List of labels where `labels[i]` is the label for document `i`

**Returns:**
- `dict` containing:
  - `leaf_purities`: Dict mapping leaf identifiers to purity scores
  - `leaf_entropies`: Dict mapping leaf identifiers to entropy scores
  - `isolation_depths`: Dict mapping class labels to isolation depths
  - `average_leaf_purity`: Average purity across all leaves
  - `average_leaf_entropy`: Average entropy across all leaves
  - `num_leaves`: Total number of leaf nodes

### `calculate_node_purity(node, labels)`

Calculate the purity of a single node (proportion of most common class).

**Parameters:**
- `node` (TreeNode): A TreeNode containing document indices
- `labels` (List[int]): List of labels

**Returns:**
- `float`: Purity score between 0 and 1

### `calculate_node_entropy(node, labels)`

Calculate the entropy of a single node.

**Parameters:**
- `node` (TreeNode): A TreeNode containing document indices
- `labels` (List[int]): List of labels

**Returns:**
- `float`: Entropy score (0 means pure, higher means more mixed)

### `calculate_all_leaf_purities(root, labels)`

Calculate node purity for all leaf nodes in the tree.

**Parameters:**
- `root` (TreeNode): The root node of the tree
- `labels` (List[int]): List of labels

**Returns:**
- `dict`: Dictionary mapping leaf identifiers to purity scores

### `calculate_all_leaf_entropies(root, labels)`

Calculate node entropy for all leaf nodes in the tree.

**Parameters:**
- `root` (TreeNode): The root node of the tree
- `labels` (List[int]): List of labels

**Returns:**
- `dict`: Dictionary mapping leaf identifiers to entropy scores

### `calculate_isolation_depth(root, labels, target_class)`

Calculate the isolation depth for a specific class.

**Parameters:**
- `root` (TreeNode): The root node of the tree
- `labels` (List[int]): List of labels
- `target_class` (int): The class label to find isolation depth for

**Returns:**
- `int` or `None`: The minimum depth at which the class is isolated, or None if never isolated

### `calculate_all_isolation_depths(root, labels)`

Calculate isolation depths for all classes in the labels.

**Parameters:**
- `root` (TreeNode): The root node of the tree
- `labels` (List[int]): List of labels

**Returns:**
- `dict`: Dictionary mapping class labels to their isolation depths

## Example Output

```python
{
    'leaf_purities': {
        <id1>: 1.0,     # Leaf 1 has perfect purity
        <id2>: 1.0,     # Leaf 2 has perfect purity
    },
    'leaf_entropies': {
        <id1>: 0.0,     # Leaf 1 has zero entropy (pure)
        <id2>: 0.0,     # Leaf 2 has zero entropy (pure)
    },
    'isolation_depths': {
        0: 1,       # Class 0 is isolated at depth 1
        1: 1,       # Class 1 is isolated at depth 1
    },
    'average_leaf_purity': 1.0,
    'average_leaf_entropy': 0.0,
    'num_leaves': 2
}
```

## Notes

- The evaluator expects labels to be aligned with document indices (i.e., `labels[i]` is the label for document `i`)
- Labels can be any integer values (0, 1, 2, ...)
- The tree root depth is 0, first level children are at depth 1, etc.
- For more examples, see `examples/demo_evaluator.py`
