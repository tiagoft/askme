# get tree filename and dataset from command line arguments
import os
import pandas as pd
import sys
import json
import numpy as np
import pydantic_ai
from tqdm import tqdm
from pathlib import Path
from load_dataset import load_dataset_sample
from askme.rtp.tree_models import TreeNode, TreePath
from askme.rtp.inference import TreeInference
from askme.rtp.generation import dict_to_path, paths_are_equal, get_random_path
from askme.askquestions.models import make_nli_model
from tqdm import tqdm
from joblib import Parallel, delayed
import askme.makequestions.api as api
import time
from pydantic import BaseModel
from pydantic_ai import Agent


class TextResponse(BaseModel):
    text: str


def generate_document_from_path(path: TreePath) -> str:
    """Generate a document string that encodes the decisions in the TreePath."""
    doc_lines = []
    for decision in path.decisions:
        ans = "Yes" if decision.decision == "entailment" else "No"
        line = f"Question: {decision.hypothesis}? Answer: {ans}"
        doc_lines.append(line)
    path_string = "\n".join(doc_lines)

    #model = api.make_gemini_model('gemini-2.5-flash-lite')
    model = api.make_ollama_model('qwen3:14b')
    prompt = f"""You are a skillful writer.
    You are writing under the following context:
    This is a newspaper article with one or two paragraphs.
    Ensure the material is coherent and relevant to the questions asked.
    You will be asked to generate material based on questions and answers.
    The first question and answer is very important, so ensure the generated material reflects it well."""

    print(prompt)

    agent = Agent(
        model=model,
        output_type=TextResponse,
        instructions=prompt,
    )
    result = agent.run_sync(f"{path_string}")
    return result.output.text, result.usage().total_tokens


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
        "--mode",
        type=str,
        required=False,
        default="test",
        help=
        "Mode of operation (default: test), can be test, gen_path or gen_naive"
    )

    parser.add_argument(
        "--n_generations",
        type=int,
        required=False,
        default=30,
        help=
        "Number of generations to use for gen_path or gen_naive modes (default: 30)"
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
    mode = args.mode
    if mode not in ["test", "gen_path", "gen_naive"]:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'test', 'gen_path', or 'gen_naive'."
        )
    n_generations = args.n_generations

    csv_filename = Path(tree_filename).with_suffix('.csv')
    if mode=='test' and csv_filename.exists():
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

    if mode == "test":
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

    elif mode == "gen_path":
        rng = np.random.default_rng()
        all_equal, all_levels, all_nlevels = [], [], []
        for _ in range(n_generations):
         
            random_path = get_random_path(tree, rng)
            try:
                print("Generating document from path...")
                document, tokens = generate_document_from_path(random_path)
                print(f"Generated document with {tokens} tokens.")
            except pydantic_ai.exceptions.ModelHTTPError:
                print("Hit my quota. Wait...")
                time.sleep(60)
                document, tokens = generate_document_from_path(random_path)
            
            print(document)
            print(random_path)
                
            predicted_node, predicted_path, predicted_label = inference_model(document)
            equal, levels = paths_are_equal(random_path, predicted_path)
            print(
                f"Generated Path vs Predicted Path Equal: {equal}, Equal Levels: {levels} ({100*levels/len(random_path.decisions):.2f}%)"
            )
            all_nlevels.append(len(random_path.decisions))
            all_equal.append(equal)
            all_levels.append(levels)
        # save all_equal, all_levels to a json file
        results = {
            "all_equal": all_equal,
            "all_levels": all_levels,
            "all_nlevels": all_nlevels,
        }
        json_filename = Path(tree_filename).with_suffix('.gen_path.json')
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
