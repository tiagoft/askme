import os
import sys
import json
import numpy as np
import pandas as pd 

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

from askme.rtp.supervised_metrics import (
    SupervisedMetric,
    NormalizedMutualInformation,
    AdjustedRandIndex,
    HomogeneityCompletenessVMeasure,
    Accuracy,
    F1Score,
    ConfusionMatrix,
)

def _pooling_fn(values):
    min_ = np.min(values)
    max_ = np.max(values)
    mu = np.mean(values)
    std = np.std(values)
    relative_std = std / mu if mu != 0 else 0
    return relative_std
    #return {'mean': mu, 'std': std, 'relative_std': relative_std, 'min': min_, 'max': max_}    

def evaluate(filename):
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
    
    supervised_metrics = [
        NormalizedMutualInformation(),
        AdjustedRandIndex(),
        HomogeneityCompletenessVMeasure(),
        Accuracy(),
        F1Score(),
        #ConfusionMatrix(),
    ]
    
    output_df = pd.DataFrame()
    for metric in unsupervised_metrics:
        result = metric(tree)
        output_df[metric.__class__.__name__] = [result]
        
    for metric in supervised_metrics:
        result = metric(tree)
        if isinstance(result, dict):
            for key, value in result.items():
                output_df[f"{metric.__class__.__name__}_{key}"] = [value]
        else:
            output_df[metric.__class__.__name__] = [result]
    return output_df

def read_input_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluates trees")

    # Define arguments
    parser.add_argument("--filename",
                        type=str,
                        nargs='+',
                        required=True,
                        help="Filename for the input data")


    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Starting tree evaluation...")
    args = read_input_arguments()
    
    output_dfs = []
    for filename in args.filename:
        output_df = evaluate(
            filename=filename
        )
        output_df['filename'] = filename
        output_dfs.append(output_df)
    final_df = pd.concat(output_dfs, ignore_index=True)
    print(final_df)