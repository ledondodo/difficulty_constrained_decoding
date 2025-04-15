# algorithms/run_task_3.py
# Author: @ledondodo

import os
import torch
from algorithms.algo_3 import (
    dynamic_beam_search,
)


def run_task_3():
    """
    Run Task 3: Dynamic Beam Search (advanced)
    Modify the beam search method from huggingface
    Pick k such that k beams probs sum up to a threshold, k clipped to bmax

    Strategies to compute dynamic k:
        - A: constant threshold decay, t = t0 * t1^i
        - B: probability mass ratio, take pmax of the probability mass
        - C: constant threshold, t = pmax
        - D1: constant threshold, softmax on flatten scores, t = pmax
        - D2: constant threshold, softmax on scores (then flatten), t = pmax
    """
    print("\n## TASK 3: Dynamic Beam Search (advanced) ##\n")

    checkpoint = "./checkpoints/smollm"
    if not os.path.exists(checkpoint):
        print("Downloading the model online...")
        checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
    model_input = "Write a story about a data scientist."

    # Configuration
    hyperparameters = {
        "N": 50,
        "pmax": 0,
        "bmax": 100,
        "use_blacklist": True,
        "max_consecutive": 20,
        "strategy": "D2",  # A, B, C, D1, D2
        "t0": 0.9,
        "t1": 0.7,
        "alpha": 0.05,
        "display": True,
    }
    print('Enter hyperparameters (for default values, in parentheses, press "enter"):')
    for key, value in hyperparameters.items():
        if not hyperparameters["strategy"] == "A" and key in ["t0", "t1"]:
            continue
        user_input = input(f"{key} (default: {value}): ")
        if user_input:
            hyperparameters[key] = eval(user_input)

    # Tasks
    dynamic_beam_search(
        checkpoint,
        model_input,
        hyperparameters=hyperparameters,  # custom
    )

    print("\n")
