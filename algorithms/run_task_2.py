# algorithms/run_task_2.py
# Author: @ledondodo

from algorithms.algo_1 import load_model
from algorithms.algo_2 import dynamic_beam_search_naive


def run_task_2():
    """
    Run Task 2: Dynamic Beam Search (naive)
    """
    print("\n## TASK 2: Dynamic Beam Search, pseudocode implementation (naive) ##\n")
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

    input = "Write a story about a data scientist."
    model, tokenizer = load_model(model_name)

    # Tasks
    dynamic_beam_search_naive(model, tokenizer, input, 9, 0.1, 100)

    print("\n")
