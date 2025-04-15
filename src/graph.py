# src/trie.py
# Author: @ledondodo
# Word graph implementation with trie-based structure

import os
import time
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import subprocess
from src.utils import make_path, MatchNotFoundError


class State:
    _id = 0

    def __init__(self, name="None"):
        self.id = State._id
        State._id += 1
        self.name = name
        self.children = {}


class Graph:
    def __init__(
        self, data, path, name, arcs=None, states=None, tokens=None, depth=None
    ):
        print("\n** Building Graph... **")
        self.data = data
        self.path = make_path(path, name)
        self.dict_states = {} if states is None else states
        self.dict_arcs = {} if arcs is None else arcs
        self.dict_tokens = {} if tokens is None else tokens
        self.key_root = 1
        self.key_finished = 2
        self.state_root = None
        self.state_finished = None
        self.time_build = 0
        self.time_display = 0
        self.depth = depth if depth else 0
        self.graph = graphviz.Digraph()

    @classmethod
    def load(cls, dir, name):
        """Load a Trie from files"""
        # Load data from pickle files
        path = os.path.join(dir, f"{name}.pkl")
        assert os.path.exists(f"{path}"), f"Directory {path} not found"
        with open(path, "rb") as f:
            saved_data = pickle.load(f)
        data = saved_data["data"]
        arcs = saved_data["arcs"]
        states = saved_data["states"]
        tokens = saved_data["tokens"]
        depth = saved_data["depth"]

        # Initialize the Trie
        trie = cls(data, dir, name, arcs, states, tokens, depth)
        trie.state_root = states["ROOT"]
        trie.state_finished = states["FINISHED"]

        return trie

    def save(self, destination=None, name=None) -> None:
        """Save the Trie to pickle files"""
        if destination and name:
            path = make_path(destination, name)
        else:
            path = self.path
        path += ".pkl"
        # Save data to pickle files
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "data": self.data,
                    "arcs": self.dict_arcs,
                    "states": self.dict_states,
                    "tokens": self.dict_tokens,
                    "depth": self.depth,
                },
                f,
            )

        print(f"Graph saved at {path}")

    def compile_pdf(self):
        """Compile the graph visualization"""
        print("\n** Compiling Graph... **")
        t = time.time()

        for state in self.dict_states.values():
            self.graph.node(str(state.id), state.name.replace(" ", "_"))
        for from_node, arcs in self.dict_arcs.items():
            for to_node, key in arcs.items():
                label = str(key) + ":" + self.dict_tokens[key].replace(" ", "_")
                self.graph.edge(str(from_node.id), str(to_node.id), label=label)

        self.graph.attr(rankdir="LR")
        self.graph.render(self.path, format="pdf", cleanup=True)
        print(f"Graph saved at {self.path}.pdf")

        self.time_display = time.time() - t

    def open_pdf(self):
        """Open the graph PDF in a new window"""
        self.compile_pdf()
        time.sleep(0.25)  # wait for the files to be created
        subprocess.Popen(["open", f"{self.path}.pdf"])

    def add_symbol(self, id: int, name: str):
        """
        Add a symbol to dict_tokens {id: name}
        inputs:
            id (int),
            name (str)
        """
        if id not in self.dict_tokens:
            self.dict_tokens[id] = name

    def add_state(self, name: str) -> int:
        """
        Add a state to dict_states {name: state}
        inputs:
            name (str)
        output:
            state (TrieNode)
        """
        if name not in self.dict_states:
            state = State(name)
            self.dict_states[name] = state
            return state
        else:
            return self.dict_states[name]

    def add_arc(self, from_state: int, to_state: int, key: int) -> None:
        """
        Add an arc to dict_arcs {from_state: {to_state: key}}
        inputs:
            from_state (TrieNode),
            to_state (TrieNode),
            key (int)
        """
        self.dict_arcs.setdefault(from_state, {}).update({to_state: key})

    def count_arcs(self):
        num_arcs = 0
        for arc in self.dict_arcs.values():
            num_arcs += len(arc)
        return num_arcs

    def metrics(self):
        print("\n** Graph Metrics **")

        print("Building Time (s):", round(self.time_build, 3))
        if self.time_display:
            print("Display Time (s):", round(self.time_display, 3))

        print("== Stats ==")
        print(
            f"Nodes: {len(self.dict_states)}, Edges: {self.count_arcs()}, Depth: {self.depth}"
        )
        print(f"Words: {len(self.data.words)}, Tokens: {len(self.dict_tokens)}")

    def valid_state(self, state, word):
        if len(word) < len(state):
            # the state is out of scope
            return False
        if word.startswith(state):
            # the state is valid
            # it exists inside a word of the list
            return True
        return False

    def recursive_build(self, state, name, word, n=0):
        # for each word, combine tokens to create new valid states
        for id, token_str in self.data.words_whitelist[word].items():
            new_name = name + token_str
            if self.valid_state(new_name, word):
                from_state = state
                if (
                    state == self.dict_states["ROOT"]
                    and token_str.startswith(tuple(self.data.spaces))
                    and token_str in self.data.spaces
                ):
                    # only allow prefix spaces, not a space as the first token
                    continue
                # new name is valid
                new_state = self.add_state(new_name)
                # add arc
                self.add_arc(from_state, new_state, id)
                # recursive call
                self.recursive_build(new_state, new_name, word, n + 1)

        # finalize state if it matches an entire word
        if name == word:
            if name[-1] in self.data.spaces:
                # From Finished_: ROOT
                new_state = self.dict_states["ROOT"]
                self.add_arc(state, new_state, self.key_root)
            else:
                # From Finished: _ROOT, BREAK, SPACE
                self.add_arc(state, self.dict_states["FINISHED"], self.key_finished)
                # Update depth
                if n > self.depth:
                    self.depth = n

    def build(self):
        assert self.data is not None, "Provide data to build the trie"
        t = time.time()

        self.state_root = self.add_state("ROOT")
        self.state_finished = self.add_state("FINISHED")
        self.add_symbol(self.key_root, "NEW")
        self.add_symbol(self.key_finished, "END")
        for id, token_str in self.data.whitelist.items():
            self.add_symbol(id, token_str)

        # Call the recursive build for each word
        for word in self.data.words:
            self.recursive_build(self.state_root, "", word)

        # FINISHED -> ROOT
        self.add_arc(self.state_finished, self.state_root, self.key_root)

        self.time_build = time.time() - t

    def recursive_match(self, seq, from_state):
        """Match a state given a sequence of tokens
        such that the sequence is a path to the state"""
        # Ending conditions:
        if len(seq) == 0:
            # 1. the whole sequence has matched
            return True, from_state
        token = seq[0].item()

        # Matching conditions: recursive calls
        if token in self.dict_arcs[from_state].values():
            # 1. token matches a transition from the current state
            next_state = [
                state for state, t in self.dict_arcs[from_state].items() if t == token
            ][0]
            is_match, state_match = self.recursive_match(seq[1:], next_state)
            if is_match:
                return is_match, state_match
        if (
            from_state == self.state_root
            and token in self.data.breaklist.keys() | self.data.spacelist.keys()
        ):
            # 2. from root, token can match a break or space, and stay in root
            is_match, state_match = self.recursive_match(seq[1:], from_state)
            return is_match, state_match
        if self.state_finished in self.dict_arcs[from_state].keys():
            # 3. finished state can be reached, continue from root
            is_match, state_match = self.recursive_match(seq, self.state_root)
            if is_match:
                return is_match, state_match

        # No match
        return False, None

    def match(self, seq):
        """Match a state given a sequence of tokens"""
        # Clip the sequence to the depth
        word = seq[-self.depth :]
        # Call the recursive function for all possible starting points
        for i in range(len(word)):
            is_match, state_match = self.recursive_match(word[i:], self.state_root)
            if is_match:
                return state_match
        # No match
        raise MatchNotFoundError(seq[-self.depth :])

    def transitions(self, from_state):
        """Get the transitions from a state"""
        transitions = set()

        for to_state, key in self.dict_arcs[from_state].items():
            if to_state == self.state_root:
                # ROOT: add next transitions + breaks + spaces
                transitions.update(self.data.breaklist.keys())
                transitions.update(self.data.spacelist.keys())
                transitions.update(self.transitions(to_state))
            elif to_state == self.state_finished:
                # FINISHED: add next transitions
                transitions.update(self.transitions(to_state))
            else:
                # Normal state: add the key (token id)
                transitions.add(key)
        return transitions
