# src/run_project.py
# Author: @ledondodo

import torch
from src.utils import get_tokenizer, get_smollm
from src.data import Data
from src.graph import Graph
from src.constrained_decoding import constrained_decoding


def get_data():
    """Load data."""
    mode = input(
        "\nChoose data source:\n1. Dataset\n2. Custom input\nEnter your choice (1 or 2): "
    ).strip()

    # 1. data source
    if mode == "1":
        custom = False
        dataset_choice = input(
            "\nChoose a dataset:\n1. CEFRJ\n2. Kaggle\nEnter your choice (1 or 2): "
        ).strip()
        if dataset_choice == "1":
            print("\nYou selected the CEFRJ dataset.")
            dataset = "CEFR-J"
        elif dataset_choice == "2":
            print("\nYou selected the Kaggle dataset.")
            dataset = "Kaggle"
        else:
            print("\nInvalid choice. Exiting...")
            return

        limit = input(
            "\nEnter the maximum number of words (press Enter for no limit): "
        ).strip()
        try:
            word_limit = int(limit) if limit else None
        except ValueError:
            print("\nInvalid input for the word limit. Exiting...")
            return

    elif mode == "2":
        custom = True
        print(
            "\nEnter your custom words one by one (press Enter without typing to finish):"
        )
        words = []
        while True:
            word = input("Enter a word: ").strip()
            if word == "":
                break
            words.append(word)
        print(f"\nYou entered the following words: {words}")
    else:
        print("\nInvalid choice. Exiting...")

    # 2. expanding data
    expand = (
        input("\nWould you like to expand the data? (y/n, default: yes): ")
        .strip()
        .lower()
    )
    if expand in ["y", ""]:
        print("\nExpanding the data...")
        expand = True
    elif expand == "n":
        print("\nProceeding without expanding the data...")
        expand = False
    else:
        print("\nInvalid choice. Exiting...")
        return

    # 3. load data
    if custom:
        data = Data(get_tokenizer(), words=words, exp=expand)
    else:
        data_path = f"data/{dataset}.csv"
        data = Data.from_csv(get_tokenizer(), data_path, size_limit=word_limit, expand=expand)

    return data


def get_graph(data):
    """Generate a graph from the data."""
    path = "./out/run"
    name = "Graph"
    graph = Graph(data, path, name)
    graph.build()
    graph.metrics()
    graph.save(path, name)

    # PDF
    generate_pdf = input(
        "\nGraph processing complete. Would you like to:\n"
        "1. Open the PDF\n"
        "2. Just compile the PDF\n"
        "Press Enter to skip: "
    ).strip()
    if generate_pdf == "1":
        print("\nGenerating and opening the PDF...")
        graph.open_pdf()
    elif generate_pdf == "2":
        print("\nGenerating the PDF without opening...")
        graph.compile_pdf()
    elif generate_pdf == "":
        print("\nSkipping PDF generation.")
    else:
        print("\nInvalid choice. Exiting...")
        return

    return graph


def generation(graph):
    """Model generation using constrained decoding."""
    # Ask for input
    input_str = input("\nEnter an input for the model: ")

    # Ask for N
    N = input("\nEnter the value for N tokens to generate (default: 25): ").strip()
    if N == "":
        N = 25
    try:
        N = int(N)
    except ValueError:
        print("\nInvalid value for N. Exiting...")
        return

    # Ask for nbeams
    nbeams = input("\nEnter the number of beams (default: 2): ").strip()
    if nbeams == "":
        nbeams = 2
    try:
        nbeams = int(nbeams)
    except ValueError:
        print("\nInvalid value for nbeams. Exiting...")
        return

    params = {
        "N": N,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "graph": graph,
        "display": True,
    }
    constrained_decoding(get_smollm(), input_str, params, nbeams=nbeams)


def run_project():
    """Run the final implementation."""
    print("\nRunning the final project implementation...")
    data = get_data()
    if data is None:
        return
    print("\nProcessing the graph...")
    graph = get_graph(data)
    print("\nConstrained decoding...")
    generation(graph)
    print("\nFinal project execution completed.")
