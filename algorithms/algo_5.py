# algorithms/algo_5.py
# Author: @ledondodo

import os
import graphviz
import string
import random


class TrieNode:
    """
    Trie Node class for states in the Finite State Machine (FSM)
    """

    _id = 0

    def __init__(self):
        self.id = TrieNode._id
        TrieNode._id += 1
        self.token_id = None
        self.children = {}
        self.is_end_of_word = False


def ask_hyperparameters(hyperparameters):
    """
    Ask the user for hyperparameters values
    leave default values if he inputs "enter" to skip

    Args:
        hyperparameters (dict): hyperparameters with default values"""
    for key, value in hyperparameters.items():
        user_input = input(f"Enter a value for {key} (default: {value}): ")
        if user_input:
            hyperparameters[key] = eval(user_input)
    print()
    return hyperparameters


def get_smollm():
    """
    Get the SmolLM model

    Returns:
        checkpoint (str): path to the model"""
    # local
    checkpoint = "./checkpoints/smollm"

    # online
    if not os.path.exists(checkpoint):
        print("Downloading the model online...")
        checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"

    return checkpoint


def extend_word_list(words):
    """
    Extend a canon list of words to include common variations

    Args:
        words (list): list of words

    Returns:
        extended_words (list): list of words and variations
    """
    extended_words = []
    for word in words:
        # Add original word, and case variations
        ## Optionally: add other variations if needed
        extended_words.append(word)
        extended_words.append(word.lower())
        extended_words.append(word.title())
        extended_words.append(word.upper())

    # Remove duplicates and return the extended list
    return list(dict.fromkeys(extended_words))


def offline_whitelist(tokenizer, words, identical=False, display=False):
    """
    From all the tokens in the tokenizer, return those that are contained in the restricted words list

    Args:
        tokenizer (transformers.PreTrainedTokenizer): tokenizer
        words (list): list of restricted words
        identical (bool): match identical words
        display (bool): display the whitelist

    Returns:
        whitelist (dict): {token: id}
    """
    # Vocab of the tokenizer
    tokens = list(set(tokenizer.get_vocab().values()))

    # Whitelist contained in the restricted words
    whitelist = {}
    for t in tokens:
        t_str = tokenizer.decode([t])
        for rw in words:
            if identical:
                if t_str == rw:
                    whitelist[t_str] = t
            else:
                if t_str in rw:
                    whitelist[t_str] = t
    whitelist = dict(sorted(whitelist.items(), key=lambda item: item[1]))
    return whitelist


def valid_state(state: string, words):
    """
    Check if a state is valid
    i.e. is the beginning of a word of the list

    Args:
        state (str): state to check
        words (list): list of words

    Returns:
        bool: True if the state is valid"""
    for word in words:
        if len(word) < len(state):
            # the state is out of scope
            continue
        if state == word[: len(state)]:
            # the state is valid
            # it exists inside a word of the list
            return True
    return False


def build_trie(root, name, words, whitelist):
    """
    Build a trie over the whitelist tokens
    Recursive call while the state is valid and update root

    Args:
        root (TrieNode): root of the FSM
        name (str): name of the state
        words (list): list of words
        whitelist (dict): {token: id}

    Returns:
        root (TrieNode): root of the FSM"""
    # try all tokens as new states
    for token in whitelist.keys():
        child_name = (name if name != "ROOT" else "") + token
        if valid_state(child_name, words):
            # child is valid
            root.children[token] = TrieNode()
            root.children[token].token_id = whitelist[token]
            build_trie(root.children[token], child_name, words, whitelist)
    if not root.children:
        # no children, end of the word
        root.is_end_of_word = True
    return root


def trie_graph(root, graph=None, parent=None, char=None):
    """
    Recursive function to build a graph for the trie, with graphviz

    Args:
        root (TrieNode): root of the FSM
        graph (graphviz.Digraph): graph to update
        parent (TrieNode): parent of the root
        char (str): character to add to the node

    Returns:
        graph (graphviz.Digraph): updated graph
    """
    if graph is None:
        graph = graphviz.Digraph(format="png")

    # Add node and edge
    node_name = str(id(root))
    graph.node(node_name, char if char else "")
    if parent is not None:
        graph.edge(str(id(parent)), node_name, label="")

    # Recursive call on children
    for char, child in root.children.items():
        trie_graph(child, graph, root, char)

    # Leaf nodes
    if root.is_end_of_word:
        graph.node(node_name + "_end", "END", style="dotted")
        graph.edge(node_name, node_name + "_end", label="")

    return graph


def visualize_trie(root, path, display):
    """
    Visualize the trie, save and show the graph

    Args:
        root (TrieNode): root of the FSM
        path (str): path to save the graph
        display (bool): display the graph
    """
    assert path is not None, "Provide a path to save the graph"
    graph = trie_graph(root, char="ROOT")
    graph.render(path)
    print(f"Graph saved at {path}.png")
    # Open the graph in a new window
    if display:
        graph.view()


def trie_FSA(words, whitelist, path, display):
    """
    Create a trie for Finte State Machine (FSM)

    Args:
        words (list): list of words
        whitelist (dict): {token: id}
        path (str): path to save the graph
        display (bool): display the graph

    Returns:
        whitelist_trie (TrieNode): root of the FSM
    """
    print("\n** Trie FSM **")
    whitelist_trie = build_trie(TrieNode(), "ROOT", words, whitelist)
    visualize_trie(whitelist_trie, path, display)
    return whitelist_trie


def match_nodes(root, sequence, display=True):
    """
    Find all nodes in the trie that match the end of the sequence

    Args:
        root (TrieNode): root of the FSM
        sequence (str): initial sequence
        display (bool): display the matches

    Returns:
        matches (dict {state id: state}): nodes that match with the end of the sequence
    """
    # get all valid states from the trie
    valid_states = {}

    def dfs(node, current_word=""):
        """Depth-first search for valid states"""
        if current_word != "":
            valid_states[node.id] = current_word
        for char, child in node.children.items():
            dfs(child, current_word + char)

    dfs(root)

    # match valid states with the end of the sequence
    matches = {}
    for state_id, state in valid_states.items():
        if sequence.endswith(state):
            matches[state_id] = state
    if display:
        print("States that match with the end of the sequence:\n", matches)
    return matches


def transition_FSM(sequence, root, display=True):
    """
    Find the new states for a sequence
    based on the trie allowed transitions

    Args:
        sequence (str): initial sequence
        root (TrieNode): root of the FSM
        display (bool): display the transitions

    Returns:
        new_states (list): list of new states"""
    if display:
        print("\n\n** Transition FSM **")
        print(f"Sequence: {sequence}\n")
    # match the sequence with the trie nodes
    matches = match_nodes(root, sequence, display)

    # get the new states, i.e. children of the matches
    new_states = []

    def dfs(node):
        """Depth-first search for children of the matches"""
        if node.id in matches.keys():
            for char, child in node.children.items():
                if char not in new_states:
                    new_states.append((char, child.token_id))
        for _, child in node.children.items():
            dfs(child)

    dfs(root)
    if not new_states:
        # no children, end of the word
        for char, child in root.children.items():
            new_states.append((char, child.token_id))
    if display:
        print("\nTransitions allowed:\n", new_states)
    return new_states


def generate_FSM(sequence, root, N):
    """
    Generate N words (randomly) using the FSM

    Args:
        sequence (str): initial sequence
        root (TrieNode): root of the FSM
        N (int): number of words
    """
    print("\n\n** Generate FSM **")
    for _ in range(N):
        new_states = transition_FSM(sequence, root, False)
        # random choice of the next state
        next_state = random.choice(new_states)
        sequence += next_state[0]
    print(sequence)
