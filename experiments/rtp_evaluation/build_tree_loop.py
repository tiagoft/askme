import toml
from pathlib import Path
from itertools import product
import subprocess
import wandb


def run():
    PROJECT_NAME = "rtp-evaluation"

    CONFIG = toml.load(Path(__file__).parent / 'experiment_config.toml')
    EXPERIMENT_PATH = Path(__file__).parent / 'build_rtp_trees.py'
    for model, strategy, nli_strategy, dataset_name in product(CONFIG['models'], CONFIG['strategies'], CONFIG['strategies_nli'], CONFIG['dataset']):
        wandb.init(project=PROJECT_NAME, config=
        {
            "model": model,
            "strategy": strategy,
            "nli_selection_strategy": nli_strategy,
            "depth": 6,
            "fraction": 0.1,
            "dataset_name": dataset_name,
        })
        
        command_line = f"python {EXPERIMENT_PATH} --model {model} --strategy {strategy} --nli_selection_strategy {nli_strategy} --depth 6 --frac 0.1 --dataset_name {dataset_name}"
        subprocess.run(command_line.split())       
        subprocess.run(["ollama", "stop", model])
        print("Stopped model")
        wandb.finish()

if __name__=="__main__":
    run()   