import argparse
from experiments.chunk_size_optimization.chunking import run_chunking_experiment
from experiments.similarity_threshold.threshold_experiment import run_threshold_experiment

TASKS = {
    "chunk_size": run_chunking_experiment,
    "similarity_threshold": run_threshold_experiment
}

def run_experiments():
    parser = argparse.ArgumentParser(description="Ejecutar experimentos RAG")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASKS.keys(),
        help="Tarea/experimento a ejecutar"
    )
    args = parser.parse_args()
    TASKS[args.task]()

if __name__ == "__main__":
    run_experiments()
