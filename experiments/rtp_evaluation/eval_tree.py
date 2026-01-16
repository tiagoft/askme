import os
import sys
import json
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from askme.rtp.tree_models import TreeNode
from askme.rtp.unsupervised_metrics import (
    UnsupervisedMetric,
    NumberOfNodes,
    TreeHeight,
    NumberOfLeafNodes,
    TreeNodeUnbalance,
    DocumentsPerLeaf,
    TreeDocumentUnbalance,
)

def _pooling_fn(values):
    min_ = np.min(values)
    max_ = np.max(values)
    mu = np.mean(values)
    std = np.std(values)
    relative_std = std / mu if mu != 0 else 0
    return {'mean': mu, 'std': std, 'relative_std': relative_std, 'min': min_, 'max': max_}    

def main(filename):
    """Main function to run the tree evaluation."""
    with open(filename, 'rb') as f:
        if filename.endswith('.pkl'):
            import pickle
            tree = pickle.load(f)
        elif filename.endswith('.json'):
            json_data = json.load(f)
            tree = TreeNode.model_validate_json(json.dumps(json_data))            
        else:
            raise ValueError("Unsupported file format. Use .pkl or .json")
    print(f"Loaded tree type: {type(tree)}")
    unsupervised_metrics = [
        NumberOfNodes(),
        TreeHeight(),
        NumberOfLeafNodes(),
        TreeNodeUnbalance(),
        DocumentsPerLeaf(pool_fn=_pooling_fn),
        TreeDocumentUnbalance(),
    ]
    for metric in unsupervised_metrics:
        result = metric(tree)
        print(f"{metric.__class__.__name__}: {result}")

def read_input_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluates trees")

    # Define arguments
    parser.add_argument("--filename",
                        type=str,
                        required=True,
                        help="Filename for the input data")


    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Starting tree evaluation...")
    args = read_input_arguments()
    main(
        filename=args.filename
    )
