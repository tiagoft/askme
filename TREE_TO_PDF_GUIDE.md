# Tree to PDF Visualization

This module provides functionality to visualize RTP (Recursive Tree Partitioning) trees as PDF documents using Graphviz.

## Features

- **Load trees from JSON**: Import previously saved TreeNode structures
- **Customizable visualization**: Control appearance, metrics display, and layout
- **Article-ready output**: High-resolution (300 DPI) PDFs suitable for publication
- **Flexible styling**: Customize fonts, colors, sizes, and orientations

## Installation

The tree-to-PDF functionality requires the `graphviz` Python package and the Graphviz system binaries:

```bash
# Python package (already in dependencies)
pip install graphviz

# System package (varies by OS)
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows: Download from https://graphviz.org/download/
```

## Quick Start

```python
from askme.rtp import TreeNode, tree_to_pdf, load_tree_from_json

# Method 1: Convert an existing tree to PDF
tree = TreeNode(documents=[0, 1, 2], question="Example question?")
tree_to_pdf(tree, output_path="my_tree")
# Creates: my_tree.pdf

# Method 2: Load from JSON and convert
tree = load_tree_from_json("saved_tree.json")
tree_to_pdf(tree, output_path="loaded_tree")
```

## API Reference

### `load_tree_from_json(json_path)`

Load a TreeNode from a JSON file.

**Parameters:**
- `json_path` (str | Path): Path to the JSON file

**Returns:**
- `TreeNode`: The loaded tree structure

**Example:**
```python
tree = load_tree_from_json("rtp_tree.json")
```

### `tree_to_graphviz(tree, **options)`

Convert a TreeNode to a Graphviz Digraph object.

**Parameters:**
- `tree` (TreeNode): The root node to visualize
- `metrics_to_display` (list[str], optional): Metric names to show in nodes
  - Examples: `['split_ratio', 'nli_calls', 'total_time_ms']`
- `font_size` (int, default=10): Font size for node labels
- `max_nodes` (int, optional): Maximum nodes to display (uses breadth-first)
- `graph_attr` (dict, optional): Graph-level attributes
  - `'rankdir'`: Direction ('TB', 'LR', 'BT', 'RL')
  - `'size'`: Dimensions in inches, e.g., '8,10'
  - `'dpi'`: Resolution (default: '300')
- `node_attr` (dict, optional): Node styling attributes
- `edge_attr` (dict, optional): Edge styling attributes

**Returns:**
- `graphviz.Digraph`: Graph object that can be rendered

**Example:**
```python
graph = tree_to_graphviz(
    tree,
    metrics_to_display=['split_ratio', 'nli_calls'],
    font_size=12,
    graph_attr={'rankdir': 'LR', 'size': '10,6'}
)
# Can then be rendered: graph.render('output', format='pdf')
```

### `tree_to_pdf(tree, output_path, **options)`

Convert a TreeNode directly to a PDF file.

**Parameters:**
- `tree` (TreeNode): The root node to visualize
- `output_path` (str | Path): Output path (without .pdf extension)
- `metrics_to_display` (list[str], optional): Metric names to display
- `font_size` (int, default=10): Font size for labels
- `max_nodes` (int, optional): Maximum nodes to display
- `graph_attr` (dict, optional): Graph attributes
- `node_attr` (dict, optional): Node attributes
- `edge_attr` (dict, optional): Edge attributes
- `cleanup` (bool, default=True): Remove intermediate DOT file

**Returns:**
- `str`: Path to the generated PDF file

**Example:**
```python
pdf_path = tree_to_pdf(
    tree,
    output_path="my_visualization",
    metrics_to_display=['split_ratio', 'nli_calls'],
    font_size=12,
    graph_attr={'rankdir': 'TB', 'size': '6,8'}
)
print(f"PDF saved to: {pdf_path}")
```

## Usage Examples

### Basic Visualization

```python
from askme.rtp import RTPBuilder, tree_to_pdf

# Build a tree
builder = RTPBuilder(use_gpu=False)
tree, metrics = builder(documents, return_metrics=True)

# Create basic PDF
tree_to_pdf(tree, output_path="basic_tree")
```

### With Split Metrics

```python
# Display specific metrics in the visualization
tree_to_pdf(
    tree,
    output_path="tree_with_metrics",
    metrics_to_display=[
        'split_ratio',
        'nli_calls', 
        'total_time_ms',
        'medoid_nli_confidence_avg'
    ],
    font_size=11
)
```

### Landscape Orientation

```python
# Create a wide, landscape-oriented visualization
tree_to_pdf(
    tree,
    output_path="landscape_tree",
    graph_attr={
        'rankdir': 'LR',  # Left to right
        'size': '10,6',   # Wide aspect ratio
    }
)
```

### Article-Ready Figure

```python
# High-quality figure for academic papers
tree_to_pdf(
    tree,
    output_path="article_figure",
    metrics_to_display=['split_ratio'],
    font_size=12,
    graph_attr={
        'rankdir': 'TB',
        'size': '6,8',    # Standard column width
        'dpi': '300',     # High resolution
    }
)
```

### Limiting Large Trees

```python
# Show only the first N nodes (breadth-first)
tree_to_pdf(
    tree,
    output_path="partial_tree",
    max_nodes=20,  # Only show 20 nodes
    font_size=10
)
```

### Custom Styling

```python
# Customize appearance
tree_to_pdf(
    tree,
    output_path="custom_style",
    font_size=11,
    node_attr={
        'shape': 'ellipse',
        'style': 'filled',
        'fillcolor': 'lightyellow',
        'fontname': 'Arial'
    },
    edge_attr={
        'color': 'darkgreen',
        'penwidth': '2'
    }
)
```

## Visualization Details

### Node Information

Each node in the tree displays:
- **Question/Hypothesis**: The decision criteria (truncated if > 60 chars)
- **Document Count**: Number of documents in the node
- **Metrics** (optional): Any requested split metrics

### Node Colors

- **Blue (lightblue)**: Internal nodes with children
- **Green (lightgreen)**: Leaf nodes (no children)

### Edge Labels

- **YES**: Documents that match the hypothesis (left child)
- **NO**: Documents that don't match (right child)

## Available Metrics

When using `metrics_to_display`, you can include any metric from the `SplitMetrics` class:

- `split_ratio`: Proportion of documents in left child
- `nli_calls`: Number of NLI model calls
- `total_time_ms`: Total processing time
- `llm_input_tokens`: Tokens sent to LLM
- `llm_output_tokens`: Tokens generated by LLM
- `faiss_search_time_ms`: Time for embedding/indexing
- `label_propagation_time_ms`: Time for label propagation
- `medoid_nli_confidence_avg`: Average NLI confidence
- `llm_request_time`: Time for LLM requests
- `nli_time`: Time for NLI processing
- `num_nodes`: Number of nodes in metrics

## Common Graph Attributes

### Layout Direction (`rankdir`)
- `'TB'`: Top to Bottom (default)
- `'LR'`: Left to Right
- `'BT'`: Bottom to Top
- `'RL'`: Right to Left

### Size (`size`)
- Format: `'width,height'` in inches
- Example: `'8,10'` for 8 inches wide, 10 inches tall
- Useful for controlling aspect ratio

### Resolution (`dpi`)
- Default: `'300'` (high quality)
- Lower for draft: `'96'` or `'150'`
- Higher for print: `'600'`

## Troubleshooting

### "dot: command not found"

The Graphviz system binaries are not installed. Install them:
- **Ubuntu/Debian**: `sudo apt-get install graphviz`
- **macOS**: `brew install graphviz`
- **Windows**: Download from https://graphviz.org/download/

### Empty or Missing PDF

Check that:
1. The tree is not empty
2. The output path is writable
3. There are no filesystem permission issues

### Large Trees Are Slow

For large trees:
1. Use `max_nodes` parameter to limit visualization
2. Consider reducing `font_size`
3. Simplify metrics with `metrics_to_display`

## Example Scripts

See the `examples/` directory for complete demonstrations:
- `demo_tree_to_pdf.py`: Simple demonstration
- `use_tree_to_pdf.py`: Comprehensive examples with various configurations

## Integration with RTP Pipeline

```python
from askme.rtp import RTPBuilder, RTPRecursion, tree_to_pdf

# Build a recursive tree
builder = RTPBuilder(use_gpu=False)
recursion = RTPRecursion(builder, min_node_size=5, max_depth=3)

# Get the tree
tree, metrics = recursion(documents)

# Save to JSON for later
with open('tree.json', 'w') as f:
    f.write(tree.model_dump_json())

# Visualize immediately
tree_to_pdf(
    tree,
    output_path="rtp_tree",
    metrics_to_display=['split_ratio', 'total_time_ms']
)

# Or load and visualize later
from askme.rtp.tree_to_pdf import load_tree_from_json
loaded_tree = load_tree_from_json('tree.json')
tree_to_pdf(loaded_tree, output_path="loaded_tree")
```
