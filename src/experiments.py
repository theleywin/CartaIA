from experiments.chunk_size_optimization.chunking import run_chunking_experiment
import json


def run_experiments():
    result = run_chunking_experiment()
    with open("./src/experiments/chunk_size_optimization/results.json", "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    run_experiments()