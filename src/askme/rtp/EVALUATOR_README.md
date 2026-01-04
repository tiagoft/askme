# Exploratory Power Evaluator

This module provides functions to evaluate the "Exploratory Power" of RTP (Retrieval-based Tree Partitioning) trees. The evaluator calculates metrics that measure how well a tree separates documents based on their labels.

## Metrics

### 1. Node Purity

Node purity measures how homogeneous a leaf node is in terms of class labels. It's calculated as:

```
Purity = 1 - Gini Impurity
```

where Gini Impurity is defined as:

```
Gini = 1 - Σ(p_i²) for each class i
```

- **Purity = 1.0**: The node is perfectly pure (all documents belong to the same class)
- **Purity = 0.0**: The node is completely impure (documents are evenly distributed across all classes)

### 2. Isolation Depth

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
print(f"Isolation depths: {results['isolation_depths']}")
```

### Individual Functions

You can also use individual functions for more granular control:

```python
from askme.rtp import (
    calculate_node_purity,
    calculate_all_leaf_purities,
    calculate_isolation_depth,
    calculate_all_isolation_depths,
)

# Calculate purity for a single node
node = TreeNode(documents=[0, 1, 2])
labels = [0, 0, 1]
purity = calculate_node_purity(node, labels)  # Returns: 0.556

# Calculate purities for all leaves
leaf_purities = calculate_all_leaf_purities(root, labels)

# Calculate isolation depth for a specific class
depth = calculate_isolation_depth(root, labels, target_class=0)

# Calculate isolation depths for all classes
all_depths = calculate_all_isolation_depths(root, labels)
```

## API Reference

### `evaluate_exploratory_power(root, labels)`

Main evaluation function that calculates both node purity and isolation depth metrics.

**Parameters:**
- `root` (TreeNode): The root node of the tree
- `labels` (List[int]): List of labels where `labels[i]` is the label for document `i`

**Returns:**
- `dict` containing:
  - `leaf_purities`: Dict mapping leaf identifiers to purity scores
  - `isolation_depths`: Dict mapping class labels to isolation depths
  - `average_leaf_purity`: Average purity across all leaves
  - `num_leaves`: Total number of leaf nodes

### `calculate_node_purity(node, labels)`

Calculate the purity of a single node.

**Parameters:**
- `node` (TreeNode): A TreeNode containing document indices
- `labels` (List[int]): List of labels

**Returns:**
- `float`: Purity score between 0 and 1

### `calculate_all_leaf_purities(root, labels)`

Calculate node purity for all leaf nodes in the tree.

**Parameters:**
- `root` (TreeNode): The root node of the tree
- `labels` (List[int]): List of labels

**Returns:**
- `dict`: Dictionary mapping leaf identifiers to purity scores

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
        0: 1.0,     # Leaf starting with document 0 has perfect purity
        3: 1.0,     # Leaf starting with document 3 has perfect purity
    },
    'isolation_depths': {
        0: 1,       # Class 0 is isolated at depth 1
        1: 1,       # Class 1 is isolated at depth 1
    },
    'average_leaf_purity': 1.0,
    'num_leaves': 2
}
```

## Notes

- The evaluator expects labels to be aligned with document indices (i.e., `labels[i]` is the label for document `i`)
- Labels can be any integer values (0, 1, 2, ...)
- The tree root depth is 0, first level children are at depth 1, etc.
- For more examples, see `examples/demo_evaluator.py`
