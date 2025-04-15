# algorithms/run_task_10.py
# Author: @ledondodo

import torch
from src.data import Data
from src.graph import Graph
from src.utils import get_tokenizer, get_smollm
from src.constrained_decoding import constrained_decoding


def run_task_10():
    print("\n## TASK 10: Graph Constrained Decoding ##")
    path = "./out/algo_10"

    load = True
    if load:
        # Load Graph
        name = "Graph_CEFR-J_A1"
        name = "Graph_Kaggle_A1"
        source = f"./data"
        print(f"Load graph from {source}")
        graph = Graph.load(source, name)
    else:
        # Custom Graph
        name = "graph_custom"
        print("Custom Graph")
        custom_words = ["i", "love", "cats", "and", "dogs"]
        data = Data(get_tokenizer(), words=custom_words)
        graph = Graph(data, path, name)
        graph.build()

    # Config
    input_str = "Reply only with the base forms of words, don’t use any inflections."
    input_str = "Describe your house."
    input_str = "Do you like dogs?"
    params = {
        "N": 25,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "graph": graph,
        "display": True,
    }

    constrained_decoding(get_smollm(), input_str, params, nbeams=2)

    print("\n")
