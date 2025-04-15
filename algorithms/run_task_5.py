# algorithms/run_task_5.py
# Author: @ledondodo

import os
from algorithms.algo_5 import (
    ask_hyperparameters,
    get_smollm,
    extend_word_list,
    offline_whitelist,
    trie_FSA,
    transition_FSM,
    generate_FSM,
)
from transformers import AutoTokenizer


def run_task_5():
    """
    Run Task 5: Offline Whitelist
    Given a list of restricted words, build a trie structure with all possible tokens combinations
    """
    print("\n## TASK 5: Offline Whitelist ##\n")

    # Configuration
    path = "out/algo_5/trie_visualization"
    hyperparameters = {
        "display": True,
        "N": 5,
    }
    hyperparameters = ask_hyperparameters(hyperparameters)

    # Tokenizer
    checkpoint = get_smollm()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Restricted words
    restricted_words = ["cat", "car", "dog"]
    restricted_words = extend_word_list(restricted_words)
    punctuation = [" ", ".", ",", "!", "?"]
    sequence = "I love cat"

    # Task
    whitelist = offline_whitelist(tokenizer, restricted_words)
    breaklist = offline_whitelist(tokenizer, punctuation, identical=True)
    FSM = trie_FSA(restricted_words, whitelist, path, hyperparameters["display"])
    new_states = transition_FSM(sequence, FSM)
    generate_FSM(sequence, FSM, hyperparameters["N"])

    print("\n")
