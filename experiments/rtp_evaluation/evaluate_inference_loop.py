from pathlib import Path
import subprocess
from joblib import Parallel, delayed

def main():
    path_to_trees = Path(__file__).parent.parent.parent / 'trees_for_ijcnn' / 'rtp_trees'
    
    # Get all json files in the directory
    tree_files = list(path_to_trees.glob('*.json'))
    print(f"Found {len(tree_files)} tree files to evaluate.")

    # Filter to only files that contain 'gpt' in their name
    tree_files = [f for f in tree_files if 'gpt' in f.name.lower()]
    print(f"Filtered to {len(tree_files)} GPT-related tree files.")

    
    # Filter out files that contain 'random' in their name
    tree_files = [f for f in tree_files if 'random' not in f.name.lower()]
    print(f"Filtered to {len(tree_files)} non-random GPT-related tree files.")
    
    
    for tree_file in tree_files:
        print(tree_file)
        if 'news' in tree_file.name:
            dataset_name = 'fancyzhx/ag_news'
        elif 'newsgroups' in tree_file.name:
            dataset_name = 'SetFit/20_newsgroups'
        elif 'wikipedia' in tree_file.name:
            dataset_name = 'wikipedia'
        elif 'bills' in tree_file.name:
            dataset_name = 'bills'
        else:
            raise ValueError(f"Unknown dataset in filename: {tree_file}")

        print(f"Evaluating tree: {tree_file.name}")
        command_line = f"python experiments/rtp_evaluation/evaluate_inference.py --tree_filename {tree_file} --dataset_name {dataset_name} --split_gt train --split test --use_cuda"
        print(f"Running command: {command_line}")
        subprocess.run(command_line.split())
        print(f"Finished evaluating tree: {tree_file.name}\n{'-'*60}\n")
    
if __name__=="__main__":
    main()