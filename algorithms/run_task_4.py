# algorithms/run_task_4.py
# Author: @ledondodo

import os
import torch
from algorithms.algo_4 import (
    words_mask,
)


def run_task_4():
    """
    Run Task 4: Words Mask
    1. Tokenize a list of words
    2. Match a sequence of tokens with (unfinished) words from the list
    3. Return the possible next tokens to complete the words
    """
    print("\n## TASK 4: Words Mask ##\n")
    checkpoint = "./checkpoints/smollm"
    if not os.path.exists(checkpoint):
        print("Downloading the model online...")
        checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"

    # Tasks
    display = True
    words_mask(checkpoint, display)

    print("\n")
