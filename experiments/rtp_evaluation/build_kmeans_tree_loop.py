import toml
from pathlib import Path
from itertools import product
import subprocess
import wandb
import torch


def run():
    PROJECT_NAME = "rtp-evaluation"

    CONFIG = toml.load(Path(__file__).parent / 'experiment_config.toml')
    EXPERIMENT_PATH = Path(__file__).parent / 'build_kmeans_trees.py'
    for model, strategy, nli_strategy, dataset_name, nli_overrides_kmeans in product(CONFIG['models'], CONFIG['strategies'], CONFIG['strategies_nli'], CONFIG['dataset'], [True, False]):
        wandb.init(project=PROJECT_NAME, config=
        {
            "model": model,
            "strategy": strategy,
            "nli_selection_strategy": nli_strategy,
            "depth": 6,
            "fraction": 0.1,
            "dataset_name": dataset_name,
            "nli_overrides_kmeans": nli_overrides_kmeans,
        })
        
        if nli_overrides_kmeans:
            nli_overrides_kmeans_str = "--nli_overrides_kmeans"
        else:
            nli_overrides_kmeans = ""
            
        torch.cuda.empty_cache()    
        command_line = f"python {EXPERIMENT_PATH} --model {model} --strategy {strategy} --nli_selection_strategy {nli_strategy} --depth 6 --frac 0.1 --dataset_name {dataset_name} {nli_overrides_kmeans_str}"
        print(command_line)
        subprocess.run(command_line.split())       
        subprocess.run(["ollama", "stop", model])
        torch.cuda.empty_cache()
        print("Stopped model")
        wandb.finish()

if __name__=="__main__":
    run()   