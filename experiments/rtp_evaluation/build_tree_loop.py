import toml
from pathlib import Path
from itertools import product
import subprocess
import wandb


def run():
    PROJECT_NAME = "rtp-evaluation"

    CONFIG = toml.load(Path(__file__).parent / 'experiment_config.toml')
    EXPERIMENT_PATH = Path(__file__).parent / 'build_trees.py'
    for model, strategy in product(CONFIG['models'], CONFIG['strategies']):
        wandb.init(project=PROJECT_NAME, config=
        {
            "model": model,
            "strategy": strategy,
            "depth": 4,
            "fraction": 0.1,
        })
        
        command_line = f"python {EXPERIMENT_PATH} --model {model} --strategy {strategy} --depth 4 --frac 0.1"
        subprocess.run(command_line.split())       
        subprocess.run(["ollama", "stop", model])
        print("Stopped model")
        wandb.finish()

if __name__=="__main__":
    run()   