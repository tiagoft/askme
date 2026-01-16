import os
import sys
import json
import numpy as np
import pandas as pd 
from tqdm import tqdm 


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

from load_dataset import load_dataset_sample

def _pooling_fn(values):
    min_ = np.min(values)
    max_ = np.max(values)
    mu = np.mean(values)
    std = np.std(values)
    relative_std = std / mu if mu != 0 else 0
    return relative_std
    #return {'mean': mu, 'std': std, 'relative_std': relative_std, 'min': min_, 'max': max_}    

def get_run_parameters(filename: str) -> dict:
    """Extract run parameters from the filename."""
    
    base = os.path.basename(filename)[:-len('.json')]
    #print(base)
    parts = base.split('_')
    #print(parts)
    params = {}
    for part in parts:
        params['dataset'] = parts[-4]
        params['llm_model_name'] = parts[-3]
        params['selection_strategy'] = parts[-2]
        params['nli_selection_strategy'] = parts[-1]
    return params

class DatasetLoader:
    """Class to load dataset samples."""
    
    def __init__(
        self,
        n_samples: int | None = 500,
        seed: int = 42,
        dataset_name: str = 'fancyzhx/ag_news'
    ):   
        texts, labels = load_dataset_sample(
            n_samples=n_samples,
            seed=seed,
            dataset_name=dataset_name
        )
        self.texts = texts
        self.labels = labels
    
    

def evaluate(filename : str, dataset: DatasetLoader) -> pd.DataFrame:
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
    #print(f"Loaded tree type: {type(tree)}")
    unsupervised_metrics = [
        #NumberOfNodes(),
        #TreeHeight(),
        #NumberOfLeafNodes(),
        TreeNodeUnbalance(),
        DocumentsPerLeaf(pool_fn=_pooling_fn),
        TreeDocumentUnbalance(),
    ]
    
    if dataset is not None:
        texts = dataset.texts
        labels = dataset.labels
        
    supervised_metrics = [
        NormalizedMutualInformation(),
        AdjustedRandIndex(),
        HomogeneityCompletenessVMeasure(),
        Accuracy(),
        F1Score(),
        #ConfusionMatrix(),
    ]
    
    output_df = pd.DataFrame(get_run_parameters(filename), index=[0])
    
    
    #print("Evaluating unsupervised metrics...")
    for metric in unsupervised_metrics:
        result = metric(tree)
        output_df[metric.__class__.__name__] = [result]
    
    if dataset is not None:
        for metric in supervised_metrics:
            result = metric(tree, dataset.labels)
            if isinstance(result, dict):
                for k, v in result.items():
                    output_df[f"{k.capitalize()}"] = [v]
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
    parser.add_argument("--dataset",
                        type=str,
                        required=False,
                        help="Dataset name (for supervised and self-supervised metrics)")


    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Starting tree evaluation...")
    args = read_input_arguments()
    print(f"Input arguments: {args}")
    if args.dataset is not None:
        print(f"Loading dataset {args.dataset}...")
        dataset = DatasetLoader(
            n_samples=None,
            dataset_name=args.dataset
        )
    else:
        dataset = None
        
    output_dfs = []
    for filename in tqdm(args.filename, desc="Evaluating files"):
        output_df = evaluate(
            filename=filename,
            dataset=dataset
        )
        output_dfs.append(output_df)
    final_df = pd.concat(output_dfs, ignore_index=True)
    print(final_df)