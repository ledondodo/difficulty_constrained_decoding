# algorithms/run_task_7.py
# Author: @ledondodo

import os
import sys
from transformers import AutoTokenizer
from src.utils import get_smollm
from algorithms.algo_7 import dataset, metrics_predictions, expand_words
from algorithms.wsm import WSM


def run_task_7():
    """
    Run Task 7: Oxford Dictionary
    Use the Oxford Dictionary dataset to build a WSM graph with Pynini implementation
    """
    print("\n## TASK 7: Oxford Dictionary ##\n")
    path = "./out/algo_7"

    # Data
    data_path = "data/CEFR-J.csv"
    words = dataset(data_path)
    breaks = [".", ",", "!", "?", "-"]
    spaces = [" ", "\n", "\t"]
    data = expand_words(words, breaks, spaces)

    pred = metrics_predictions(path, 5000, "quad")
    return
    # Tokenizer
    checkpoint = get_smollm()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Graph
    size = 10
    data = {
        "words": words[:size],
        "breaks": [".", ",", "!", "?", "-"],
        "spaces": [" ", "\n", "\t"],
    }
    oxford = WSM()
    oxford.build(data, tokenizer)
    oxford.save(f"oxford_{size}", destination=path)
    oxford.compile_graph()

    # Results and metrics
    oxford.metrics()
    pred = metrics_predictions(path, len(words), "quad")

    # Compute PDF
    source = "./data/job_7"
    if os.path.exists(f"{source}/oxford_A1"):
        oxford.compile_pdf()
        wsm = WSM.load(f"oxford_A1", source)
        dest = "./data/job_7"
        wsm.save("oxford_A1", destination=dest)
        wsm.compile_pdf()
    else:
        print("PDF not created. Missing source folder.")

    print("\n")
