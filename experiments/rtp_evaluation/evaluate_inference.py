# get tree filename and dataset from command line arguments
import os
import pandas as pd
import sys
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from load_dataset import load_dataset_sample
from askme.rtp.tree_models import TreeNode
from askme.rtp.inference import TreeInference
from askme.rtp.generation import dict_to_path, paths_are_equal
from askme.askquestions.models import make_nli_model
from tqdm import tqdm
from joblib import Parallel, delayed

def read_input_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RTP tree inference")

    # Define arguments
    parser.add_argument("--tree_filename",
                        type=str,
                        required=True,
                        help="Path to the RTP tree JSON file")
    parser.add_argument("--dataset_name",
                        type=str,
                        required=True,
                        help="Name of the dataset to evaluate on")
    parser.add_argument(
        "--n_samples_gt",
        type=int,
        required=False,
        default=None,
        help="Number of samples to use for ground truth (default: all samples)"
    )
    parser.add_argument(
        "--split_gt",
        type=str,
        required=False,
        default="test",
        help="Dataset split to use for ground truth(default: test)")
    parser.add_argument(
        "--n_samples",
        type=int,
        required=False,
        default=None,
        help="Number of samples to use for evaluation (default: all samples)")
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="Dataset split to use for evaluation (default: test)")
    parser.add_argument(
        "--nli_model_name",
        type=str,
        required=False,
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        help=
        "Name of the NLI model to use (default: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)"
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Whether to use CUDA for NLI model (default: False)")
    return parser.parse_args()


def load_assets(tree_filename: str, dataset_name: str,
                n_samples_gt: int | None, split_gt: str, n_samples: int | None,
                split: str, model_name: str, use_cuda: bool):
    # Read the RTP tree from the JSON file
    #print("Reading tree...")
    with open(tree_filename, 'r') as f:
        tree_json = f.read()
        tree = TreeNode.model_validate_json(tree_json)

    #print("Loading NLI model...")
    model, tokenizer = make_nli_model(model_name)

    #print("Loading dataset samples...")
    text_gt, labels_gt = load_dataset_sample(
        n_samples=n_samples_gt,
        seed=42,
        dataset_name=dataset_name,
        split=split_gt,
    )

    if split_gt != split or n_samples_gt != n_samples:
        text_eval, labels_eval = load_dataset_sample(
            n_samples=n_samples,
            seed=42,
            dataset_name=dataset_name,
            split=split,
        )
    else:
        text_eval, labels_eval = text_gt, labels_gt

    #print("Setting up inference model...")
    if use_cuda:
        model.to('cuda')
        device = 'cuda:0'
    else:
        device = 'cpu'

    inference_model = TreeInference(
        tree_root=tree,
        nli_model=model,
        nli_tokenizer=tokenizer,
        device=device,
        ground_truth_labels=labels_gt,
    )

    return tree, inference_model, text_eval, labels_eval


def main():
    args = read_input_arguments()
    tree_filename = args.tree_filename
    dataset_name = args.dataset_name
    n_samples = args.n_samples
    split = args.split
    n_samples_gt = args.n_samples_gt
    split_gt = args.split_gt
    model_name = args.nli_model_name
    use_cuda = args.use_cuda

    csv_filename = Path(tree_filename).with_suffix('.csv')
    if csv_filename.exists():
        print(f"CSV file {csv_filename} already exists. Skipping evaluation.")
        return

    tree, inference_model, text_eval, labels_eval = load_assets(
        tree_filename,
        dataset_name,
        n_samples_gt,
        split_gt,
        n_samples,
        split,
        model_name,
        use_cuda,
    )
    def infer_one(document, true_label):
        predicted_node, predicted_path, predicted_label = inference_model(document)
        return predicted_label

    predicted_labels = []
    for text in tqdm(
            zip(text_eval, labels_eval),
            desc="Evaluating samples",
            total=len(text_eval),
    ):
        document, true_label = text
        predicted_node, predicted_path, predicted_label = inference_model(
            document)
        predicted_labels.append(predicted_label)

    # Evaluate accuracy
    predicted_labels = np.array(predicted_labels)
    labels_eval = np.array(labels_eval)
    # Use sklearn to calculate accuracy
    from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
    accuracy = accuracy_score(labels_eval, predicted_labels)
    f1 = f1_score(labels_eval, predicted_labels, average='macro')
    nmi = normalized_mutual_info_score(labels_eval, predicted_labels)
    ari = adjusted_rand_score(labels_eval, predicted_labels)
    # Change filename extension to .csv
    csv_filename = Path(tree_filename).with_suffix('.csv')
    # Save results to csv
    df = pd.DataFrame({
        'Tree Filename': [tree_filename],
        'Dataset': [dataset_name],
        'Accuracy': [accuracy],
        'F1 (macro)': [f1],
        'NMI': [nmi],
        'ARI': [ari],
    })
    df.to_csv(csv_filename, index=False)
    
    print(f"Filename: {tree_filename}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")


if __name__ == "__main__":
    main()
