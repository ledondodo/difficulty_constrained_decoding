# algorithms/run_task_9.py
# Author: @ledondodo

import os
from src.data import Data
from src.graph import Graph
from src.utils import get_tokenizer


def run_task_9():
    """
    Run Task 9: Trie Structure
    Trie structure implementation
    Generate a graph from a vocabulary
    """
    print("\n## TASK 9: Trie Structure ##")
    path = "./out/algo_9"
    custom = False

    if custom:
        # Custom Data
        custom_words = ["cat", "car", "coca", "dog"]
        data = Data(get_tokenizer(), words=custom_words, exp=False)
        name = "cat"
        save = False
    else:
        # Load Data
        name = "CEFR-J"
        name = "Kaggle"
        data_path = f"data/{name}.csv"
        name = f"Graph_{name}_A1"
        data = Data.from_csv(get_tokenizer(), data_path, size_limit=None)
        save = True

    # Graph
    graph = Graph(data, path, name)
    graph.build()

    # Save
    path_data = "./data" if save else path
    graph.save(path_data, name)

    # PDF
    if custom:
        graph.open_pdf()
    elif not save:
        graph.compile_pdf()

    # Metrics
    graph.metrics()

    print("\n")
