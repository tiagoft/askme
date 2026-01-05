"""Module for converting RTP trees to PDF visualizations using Graphviz."""

import json
from pathlib import Path
from typing import Optional, Union, List
import graphviz
from .tree_models import TreeNode, SplitMetrics


def load_tree_from_json(json_path: Union[str, Path]) -> TreeNode:
    """
    Load a TreeNode from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing the tree
        
    Returns:
        TreeNode: The loaded tree structure
        
    Example:
        >>> tree = load_tree_from_json("rtp_tree.json")
    """
    with open(json_path, 'r') as f:
        tree_dict = json.load(f)
    return TreeNode.model_validate(tree_dict)


def tree_to_graphviz(
    tree: TreeNode,
    metrics_to_display: Optional[List[str]] = None,
    font_size: int = 10,
    max_nodes: Optional[int] = None,
    graph_attr: Optional[dict] = None,
    node_attr: Optional[dict] = None,
    edge_attr: Optional[dict] = None,
) -> graphviz.Digraph:
    """
    Convert a TreeNode to a Graphviz Digraph for visualization.
    
    Args:
        tree: The root TreeNode to visualize
        metrics_to_display: List of metric names to display in nodes (e.g., 
            ['split_ratio', 'nli_calls', 'total_time_ms']). If None, only basic
            node info is shown.
        font_size: Font size for node labels (default: 10)
        max_nodes: Maximum number of nodes to display. If None, all nodes are shown.
            If the tree has more nodes, only the first max_nodes are shown (breadth-first).
        graph_attr: Additional graph attributes (e.g., {'rankdir': 'TB', 'size': '8,10'}).
            Common attributes:
            - 'rankdir': 'TB' (top-bottom), 'LR' (left-right), 'BT', 'RL'
            - 'size': e.g., '8,10' for width,height in inches (aspect ratio)
            - 'ratio': 'fill', 'compress', 'auto', or a float value
        node_attr: Additional node attributes (e.g., {'shape': 'box', 'style': 'rounded'})
        edge_attr: Additional edge attributes (e.g., {'color': 'blue'})
        
    Returns:
        graphviz.Digraph: Graphviz graph object that can be rendered to PDF
        
    Example:
        >>> graph = tree_to_graphviz(
        ...     tree,
        ...     metrics_to_display=['split_ratio', 'nli_calls'],
        ...     font_size=12,
        ...     max_nodes=20,
        ...     graph_attr={'rankdir': 'TB', 'size': '8,10'}
        ... )
        >>> graph.render('output', format='pdf', cleanup=True)
    """
    # Default graph attributes for article-quality figures
    default_graph_attr = {
        'rankdir': 'TB',  # Top to bottom
        'dpi': '300',  # High resolution for articles
    }
    if graph_attr:
        default_graph_attr.update(graph_attr)
    
    # Default node attributes
    default_node_attr = {
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': 'lightblue',
        'fontsize': str(font_size),
    }
    if node_attr:
        default_node_attr.update(node_attr)
    
    # Default edge attributes
    default_edge_attr = {
        'fontsize': str(max(8, font_size - 2)),
    }
    if edge_attr:
        default_edge_attr.update(edge_attr)
    
    # Create the graph
    graph = graphviz.Digraph(
        graph_attr=default_graph_attr,
        node_attr=default_node_attr,
        edge_attr=default_edge_attr,
    )
    
    # Track number of nodes added
    nodes_added = 0
    
    # Use breadth-first traversal to add nodes
    queue = [(tree, 'node_0', None, None)]  # (node, node_id, parent_id, edge_label)
    node_counter = 0
    
    while queue and (max_nodes is None or nodes_added < max_nodes):
        current_node, node_id, parent_id, edge_label = queue.pop(0)
        
        # Create node label with basic info
        label_parts = []
        
        # Add question/hypothesis if present
        if current_node.question:
            # Truncate long questions for readability
            question = current_node.question
            if len(question) > 60:
                question = question[:57] + "..."
            label_parts.append(f"Q: {question}")
        
        # Add number of documents
        n_docs = len(current_node.documents)
        label_parts.append(f"Docs: {n_docs}")
        
        # Add metrics if requested
        if metrics_to_display and current_node.metrics:
            for metric_name in metrics_to_display:
                if hasattr(current_node.metrics, metric_name):
                    value = getattr(current_node.metrics, metric_name)
                    # Format metric value appropriately
                    if isinstance(value, float):
                        if metric_name.endswith('_time_ms') or metric_name.endswith('_time'):
                            label_parts.append(f"{metric_name}: {value:.1f}ms")
                        elif metric_name in ['split_ratio', 'medoid_nli_confidence_avg']:
                            label_parts.append(f"{metric_name}: {value:.3f}")
                        else:
                            label_parts.append(f"{metric_name}: {value:.2f}")
                    else:
                        label_parts.append(f"{metric_name}: {value}")
        
        # Create the label
        label = "\\n".join(label_parts)
        
        # Determine node color based on whether it's a leaf
        is_leaf = current_node.left is None and current_node.right is None
        node_fillcolor = 'lightgreen' if is_leaf else 'lightblue'
        
        # Add node to graph
        graph.node(node_id, label=label, fillcolor=node_fillcolor)
        nodes_added += 1
        
        # Add edge from parent if exists
        if parent_id is not None and edge_label is not None:
            graph.edge(parent_id, node_id, label=edge_label)
        
        # Add children to queue if not at max_nodes
        if max_nodes is None or nodes_added < max_nodes:
            if current_node.left is not None:
                node_counter += 1
                queue.append((current_node.left, f'node_{node_counter}', node_id, 'YES'))
            
            if current_node.right is not None:
                node_counter += 1
                queue.append((current_node.right, f'node_{node_counter}', node_id, 'NO'))
    
    return graph


def tree_to_pdf(
    tree: TreeNode,
    output_path: Union[str, Path],
    metrics_to_display: Optional[List[str]] = None,
    font_size: int = 10,
    max_nodes: Optional[int] = None,
    graph_attr: Optional[dict] = None,
    node_attr: Optional[dict] = None,
    edge_attr: Optional[dict] = None,
    cleanup: bool = True,
) -> str:
    """
    Convert a TreeNode to a PDF file using Graphviz.
    
    Args:
        tree: The root TreeNode to visualize
        output_path: Path where the PDF should be saved (without extension)
        metrics_to_display: List of metric names to display in nodes
        font_size: Font size for node labels (default: 10)
        max_nodes: Maximum number of nodes to display
        graph_attr: Additional graph attributes for customization
        node_attr: Additional node attributes for customization
        edge_attr: Additional edge attributes for customization
        cleanup: Whether to remove intermediate DOT file (default: True)
        
    Returns:
        str: Path to the generated PDF file
        
    Example:
        >>> tree = load_tree_from_json("rtp_tree.json")
        >>> pdf_path = tree_to_pdf(
        ...     tree,
        ...     output_path="tree_visualization",
        ...     metrics_to_display=['split_ratio', 'nli_calls'],
        ...     font_size=12,
        ...     graph_attr={'rankdir': 'LR', 'size': '10,8'}
        ... )
        >>> print(f"PDF saved to {pdf_path}")
    """
    # Convert tree to Graphviz graph
    graph = tree_to_graphviz(
        tree=tree,
        metrics_to_display=metrics_to_display,
        font_size=font_size,
        max_nodes=max_nodes,
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )
    
    # Render to PDF
    output_path_str = str(output_path)
    result = graph.render(output_path_str, format='pdf', cleanup=cleanup)
    
    return result
