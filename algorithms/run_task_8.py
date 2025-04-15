# algorithms/run_task_8.py
# Author: @ledondodo

import os
import sys
import torch
from transformers import AutoTokenizer
from src.utils import get_smollm
from algorithms.wsm import WSM
from algorithms.algo_7 import expand_words
from algorithms.algo_8 import constrained_decoding


def run_task_8():
    """
    Run Task 8: Constrained Decoding
    Generate with constrained decoding from the Pynini implementation
    """
    print("\n## TASK 8: Constrained Decoding ##\n")

    load = True
    if load:
        # Load WSM
        source = "./data/wsm"
        print(f"Load WSM from {source}")
        wsm = WSM.load("words_exp", source)
        wsm.metrics()
    else:
        # Custom WSM
        print("Custom WSM")
        checkpoint = get_smollm()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        words = [
            "the",
            "a",
            "not",
            "cat",
            "dog",
            "guy",
            "father",
            "son",
            "house",
            "tv",
            "apple",
            "is",
            "are",
            "have",
            "has",
            "want",
            "good",
            "bad",
            "big",
            "small",
            "best",
            "worst",
            "terrific",
        ]
        breaks = [".", ",", "!", "?", "-"]
        spaces = [" ", "\n", "\t"]
        data = expand_words(words, breaks, spaces)
        wsm = WSM()
        wsm.build(data, tokenizer)
        wsm.metrics()

    print("Words:", wsm.words)
    print("Whitelist:", wsm.whitelist)
    print("Breaklist:", wsm.breaklist)
    print("Spacelist:", wsm.spacelist)
    print()

    # Config
    checkpoint = get_smollm()
    input_str = "Generate a story"
    params = {
        "N": 20,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "wsm": wsm,
        "display": True,
    }

    # Generate
    constrained_decoding(checkpoint, input_str, params)

    print("\n")
