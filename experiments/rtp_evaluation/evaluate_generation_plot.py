import json
from pathlib import Path
import pandas as pd


def read_input_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Plot figures for RTP generation evaluation results.")

    # Define arguments
    parser.add_argument("--tree_filename",
                        type=str,
                        nargs='+',
                        required=True,
                        help="Path to the RTP tree JSON file")
    
    return parser.parse_args()

def run():
    args = read_input_arguments()
    filenames = args.tree_filename
    
    for tree_filename in filenames:
        
        suffix = ".gen_path.json"
        json_filename = Path(tree_filename).with_suffix(suffix)
        if not json_filename.exists():
            raise FileNotFoundError(f"Expected file {json_filename} does not exist.")
        print("Processing suffix:", suffix)
        with open(json_filename, 'r') as f:
            results = json.load(f)
        df_path = pd.DataFrame(results)
        
        suffix = ".gen_naive.json"
        json_filename = Path(tree_filename).with_suffix(suffix)
        if not json_filename.exists():
            raise FileNotFoundError(f"Expected file {json_filename} does not exist.")
        print("Processing suffix:", suffix)
        with open(json_filename, 'r') as f:
            results = json.load(f)
        df_naive = pd.DataFrame(results)

        df_naive['avg_correct_levels'] = df_naive['all_levels'] / df_naive['all_nlevels']
        df_path['avg_correct_levels'] = df_path['all_levels'] / df_path['all_nlevels']    
        levels = [df_path['avg_correct_levels'].mean(), df_naive['avg_correct_levels'].mean()]

        accuracies = [df_path['all_equal'].mean(), df_naive['all_equal'].mean()]
        print("Levels comparison (average node depth)")
        print(levels)
        print("Accuracies comparison (P_path):")
        print(accuracies)
        print("Accuracies comparison (P_node):")
        print([a**(1/5) for a in accuracies])
        
            

if __name__ == "__main__":
    run()