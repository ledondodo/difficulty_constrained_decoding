# algorithms/run_task_6.py
# Author: @ledondodo

import os
import sys
from transformers import AutoTokenizer
from algorithms.algo_5 import get_smollm
from algorithms.algo_6 import simple_graph
from algorithms.wsm import WSM


def run_task_6():
    """
    Run Task 6: Pynini FSA
    Use Pynini to build a Finite State Automaton (FSA) for a list of words
    """
    print("\n## TASK 6: Pynini FSA ##\n")
    path = "./out/algo_6"
    store_path = "./data/wsm"

    # Tokenizer
    checkpoint = get_smollm()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokens
    breaks = [".", ",", "!", "?", "-"]
    spaces = [" ", "\n", "\t"]

    # 1. simple example
    print("** Simple Graph **")
    simple = simple_graph()
    simple.save("simple", destination=path)
    simple.compile_pdf()

    # 2. words example
    print("** Word Graph **")
    data = {"words": ["cat", "dog"], "breaks": breaks, "spaces": spaces}
    wsm = WSM()
    wsm.build(data, tokenizer)
    wsm.save("words", destination=path)
    wsm.save("words", destination=store_path)
    wsm.compile_pdf()

    # 3. complex wsm
    print("** WSM Graph **")
    data = {
        "words": [
            "cat",
            "dog",
            "on",
            "the",
            "mat",
            "airplaine",
            "race",
            "machine",
            "engineer",
            "data",
            "scientist",
            "machine",
            "learning",
            "artificial",
            "intelligence",
        ],
        "breaks": breaks,
        "spaces": spaces,
    }
    data["words"] += [w + "." for w in data["words"]] + [w + " " for w in data["words"]]
    wsm = WSM()
    wsm.build(data, tokenizer)
    wsm.save("wsm", destination=path)
    wsm.compile_pdf()

    print("\n")
