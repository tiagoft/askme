import toml
from pathlib import Path
import json
from askme.rtp import TreeNode, SplitMetrics, evaluate_exploratory_power


def process_file(input_fn: str):
    input_path = Path(input_fn)
    with open(input_path, 'r') as f:
        json_data = f.read()
    loaded_node = TreeNode.model_validate_json(json_data)

    metrics = aggregate_metrics(loaded_node)
    n_nodes = count_nodes(loaded_node)
    n_leaves = count_leaves(loaded_node)
    metrics.medoid_nli_confidence_avg /= (n_nodes-n_leaves)
    metrics.split_ratio /= (n_nodes-n_leaves)
    
    
    print("Aggregated Tree Metrics:")
    print(metrics.model_dump_json(indent=4))


def run():
    args = read_input_arguments()
    inputs = args.input

    for input_fn in inputs:
        process_file(input_fn)
    

def count_leaves(node: TreeNode) -> int:
    """Recursively count the number of leaf nodes in the tree."""
    if node.left is None and node.right is None:
        return 1
    count = 0
    if node.left is not None:
        count += count_leaves(node.left)
    if node.right is not None:
        count += count_leaves(node.right)
    return count

def count_nodes(node: TreeNode) -> int:
    """Recursively count the number of nodes in the tree."""
    count = 1  # Count the current node
    if node.left is not None:
        count += count_nodes(node.left)
    if node.right is not None:
        count += count_nodes(node.right)
    return count

def aggregate_metrics(node: TreeNode) -> SplitMetrics:
    """Recursively aggregate SplitMetrics from the tree."""
    if node.metrics is not None:
        current_metrics = node.metrics
    else:
        current_metrics = SplitMetrics()

    if node.left is not None:
        left_metrics = aggregate_metrics(node.left)
        current_metrics += left_metrics

    if node.right is not None:
        right_metrics = aggregate_metrics(node.right)
        current_metrics += right_metrics
    
    return current_metrics
    
def read_input_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trees")


    # Define arguments
    parser.add_argument("--input",
                        type=str,
                        nargs="+",
                        required=True,
                        help="Input file path")

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__=="__main__":
    run()
    