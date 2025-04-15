# src/wsm.py
# Author: @ledondodo
# Word graph implementation with pynini

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../libs/gcd"))
from gcd.inference.automata import FSA
import pynini
import subprocess
import pickle
import time


def find_tokens(tokenizer, words, identical=False):
    """Offline token finder:
    From all the tokens in the tokenizer,
    return those that are contained in the list of words
    return dict: {token: id}"""
    # Vocab of the tokenizer
    vocab = list(set(tokenizer.get_vocab().values()))

    # Tokens contained in the restricted words
    tokens = {}
    for t in vocab:
        t_str = tokenizer.decode([t])
        for w in words:
            if identical:
                if t_str == w:
                    tokens[t_str] = t
            else:
                if t_str in w:
                    tokens[t_str] = t
    tokens = dict(sorted(tokens.items(), key=lambda item: item[1]))
    return tokens


class MatchNotFoundError(Exception):
    """Raised when a required match is not found."""

    def __init__(self, target):
        super().__init__(f"No match found for: {target}")


class WSM(FSA):
    """Word State Machine, for constrained decoding"""

    def config(self, data):
        """Config from data"""
        self.words = data["words"]
        self.whitelist = data["whitelist"]
        self.breaklist = data["breaklist"]
        self.spacelist = data["spacelist"]
        self.vocab_size = data["vocab_size"]

    def __init__(self, fst=None, sym=None, states=None, data=None, path=None):
        super().__init__()
        if fst:
            self.fst = fst
        if sym:
            self.symbol_table = sym
        if states:
            self.state_names = states
        else:
            self.state_names = pynini.SymbolTable()
        if data:
            self.config(data)
        else:
            self.words = []
            self.whitelist = {}
            self.breaklist = {}
            self.spacelist = {}
            self.vocab_size = 0
        if path:
            self.path = path
        else:
            self.path = None
        self.key_start = 1
        self.key_break = 2
        self.key_space = 3
        self.time = 0
        self.depth = 0

    @classmethod
    def load(cls, name, dir):
        """Load a WSM from files .fst and .sym"""
        dir = f"{dir}/{name}"
        fst = pynini.Fst.read(f"{dir}/{name}.fst")
        sym = pynini.SymbolTable.read(f"{dir}/{name}_symbols.sym")
        states = pynini.SymbolTable.read(f"{dir}/{name}_states.sym")
        with open(f"{dir}/{name}_data.pkl", "rb") as f:
            data = pickle.load(f)
        wsm = cls(fst, sym, states, data)
        return wsm

    def save(self, name: str, destination=None) -> None:
        """Save WSM to file"""
        if destination:
            self.path = f"{destination}/{name}/{name}"
        assert self.path, "Add a destination"
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        self.fst.write(f"{self.path}.fst")
        self.symbol_table.write(f"{self.path}_symbols.sym")
        self.state_names.write(f"{self.path}_states.sym")
        data = {
            "vocab_size": self.vocab_size,
            "words": self.words,
            "whitelist": self.whitelist,
            "breaklist": self.breaklist,
            "spacelist": self.spacelist,
        }
        with open(f"{self.path}_data.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"WSM saved at {os.path.dirname(self.path)}\n")

    def compile_graph(self):
        """Compile graph: DOT (Graphviz)"""
        assert self.path, "First save the WSM"
        self.fst.draw(
            source=f"{self.path}.gv",
            isymbols=None,
            osymbols=self.symbol_table,
            ssymbols=self.state_names,
            portrait=True,
        )

    def compile_pdf(self):
        """Compile PDF"""
        self.compile_graph()
        cmd_args = ["dot", "-Tpdf", f"{self.path}.gv", "-o", f"{self.path}.pdf"]
        subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def open_pdf(self):
        """Open PDF"""
        self.compile_pdf()
        time.sleep(0.25)  # wait for the files to be created
        subprocess.Popen(["open", f"{self.path}.pdf"])

    def add_state(self, name=None) -> int:
        """Add a state and optionally assign a name"""
        assert not self._compiled
        if not self.state_names.member(name):
            # the state does not exist yet
            state_id = self.fst.add_state()
            if name:
                self.state_names.add_symbol(name, state_id)
            return state_id
        else:
            # the state has already been created
            state_id = self.state_names.find(name)
            return state_id

    def add_arc(self, from_state: int, to_state: int, key: int) -> None:
        """Add an arc between two states
        check for existence and uniqueness"""
        assert not self._compiled
        count = 0
        unique = True
        for a in self.fst.arcs(from_state):
            # existing arcs from state
            count += 1
            if a.nextstate == to_state:
                unique = False
        if not count or unique:
            # add only unique arcs (avoid duplicates)
            # or if there is no arc yet
            self.fst.add_arc(from_state, pynini.Arc(key, key, None, to_state))

    def count_arcs(self, state, n=0, visited=[]):
        """Recursive function that propagate through the graph,
        and counts arcs leaving a state"""
        visited.append(state)
        for a in self.fst.arcs(state):
            n += 1
            if a.nextstate == state or a.nextstate == self.fst.start():
                # ending condition: self loop, or end state
                continue
            elif a.nextstate in visited:
                # already visited
                continue
            else:
                n, visited = self.count_arcs(a.nextstate, n, visited)
        return n, visited

    def spread(self, state, n=0, visited=[]):
        """Recursive function that propagate through the graph,
        and compute the depth of the graph"""
        n += 1
        visited.append(state)
        for a in self.fst.arcs(state):
            if (
                a.nextstate == state
                or a.nextstate == self.fst.start()
                or a.nextstate in visited
            ):
                # ending condition: self loop, or end state
                if n > self.depth:
                    self.depth = n
                return n
            self.spread(a.nextstate, n)

    def metrics(self):
        print("** WSM Metrics **")
        from_state = self.fst.start()
        num_arcs, _ = self.count_arcs(from_state)
        self.spread(from_state)
        print(f"size: {len(self.words)}\t\t\t depth: {self.depth}")
        print(f"building time: {self.time:.3f} s\t nodes: {self.fst.num_states()}")
        print(f"unique tokens: {self.symbol_table.num_symbols()}\t edges: {num_arcs}\n")

    def valid_state(self, state):
        """Check if a state is valid
        i.e. is the beginning of a word of the list"""
        for word in self.words:
            if len(word) < len(state):
                # the state is out of scope
                continue
            if state == word[: len(state)]:
                # the state is valid
                # it exists inside a word of the list
                return True
        return False

    def finished_state(self, name):
        """A state is finished it it matches an entire word"""
        for word in self.words:
            if name == word:
                return True
        return False

    def recursive_build(self, state, name):
        """Build a WSM over the whitelist tokens,
        Recursive call while the state is valid"""
        # for each token, create a new state that match a word
        for token_str in sorted(self.whitelist.keys()):
            new_name = name + token_str
            if self.valid_state(new_name):
                # new name is valid
                new_state = self.add_state(name=new_name)
                # add arc
                id = self.symbol_table.find(token_str)
                self.add_arc(state, new_state, id)
                # recursive call
                self.recursive_build(new_state, new_name)

        # finalize state if it matches an entire word
        if self.finished_state(name):
            # only states ending with a space can loop to root
            for sp in self.spacelist:
                if name.endswith(sp):
                    new_state = self.state_names.find("ROOT")
                    self.add_arc(state, new_state, self.key_start)
                    return
            # any state can add a break or a space
            new_state = self.state_names.find("BREAK")
            self.add_arc(state, new_state, self.key_break)
            new_state = self.state_names.find("SPACE")
            self.add_arc(state, new_state, self.key_space)
            # additionnaly link to states that start with a space
            for token_str in sorted(self.whitelist.keys()):
                ok = False
                for sp in self.spacelist:
                    if token_str.startswith(sp) and len(token_str) > len(sp):
                        # token starts with a space and is not only a space
                        ok = True
                if ok:
                    new_name = token_str
                    new_state = self.add_state(name=new_name)
                    # add arc
                    id = self.symbol_table.find(token_str)
                    self.add_arc(state, new_state, id)

    def build(self, data, tokenizer):
        """Build a WSM for constrained decoding on data
        data: dict (words, breaks, spaces)"""
        # record time
        t = time.time()

        # get specific tokens from the tokenizer
        data["whitelist"] = find_tokens(tokenizer, data["words"])
        data["breaklist"] = find_tokens(tokenizer, data["breaks"], identical=True)
        data["spacelist"] = find_tokens(tokenizer, data["spaces"], identical=True)
        data["vocab_size"] = tokenizer.vocab_size

        # config wsm
        self.config(data)
        start_state = self.add_state(name="ROOT")
        break_state = self.add_state(name="BREAK")
        space_state = self.add_state(name="SPACE")
        self.set_start(start_state)
        self.set_final(space_state)
        self.add_symbol("NEW", self.key_start)
        self.add_symbol("BREAK", self.key_break)
        self.add_symbol("SPACE", self.key_space)
        for token, id in self.whitelist.items():
            self.add_symbol(token, id)

        # build wsm
        self.recursive_build(start_state, "")

        # finalize wsm
        self.add_arc(start_state, break_state, self.key_break)  # direct break from root
        self.add_arc(start_state, space_state, self.key_space)  # direct break from root
        self.add_arc(
            break_state, break_state, self.key_break
        )  # break can loop on itself
        self.add_arc(
            break_state, space_state, self.key_space
        )  # break can be followed by a space
        self.add_arc(
            space_state, start_state, self.key_start
        )  # space can go back to root

        # record time
        self.time = time.time() - t

    def match(self, seq):
        """Match a state given a sequence of tokens"""
        # graph depth is the maximum number of tokens in a word
        word = seq[-self.depth :]

        # try to exact match any word of size smaller than depth
        while len(word) > 0:
            # start from the root
            state = self.get_start()
            for i, token in enumerate(word):
                token_match = False
                for a in self.fst.arcs(state):
                    if a.ilabel == self.key_break:
                        # 1. break token
                        if token.item() in self.breaklist.values():
                            state = self.state_names.find("BREAK")
                            token_match = True
                    elif a.ilabel == self.key_space:
                        # 2. space token (ending state)
                        if i < len(word) - 1:
                            # not the last token
                            break
                        elif token.item() in self.spacelist.values():
                            # match a space token: update state
                            state = self.state_names.find("SPACE")
                            token_match = True
                    elif a.ilabel == token.item():
                        # 3. standard token
                        # match: update state
                        state = a.nextstate
                        token_match = True
                        break
                if not token_match:
                    # no match: break
                    break
                if i == len(word) - 1:
                    # every token in the word have matched
                    return state
            # try to match a smaller word
            word = word[1:]
        raise MatchNotFoundError(seq[-self.depth :])

    def transitions(self, state):
        """Get the transitions from a state"""
        transitions = []
        for a in self.fst.arcs(state):
            if a.ilabel == self.key_start:
                # transition to root = new word: any first token (transitions from root)
                transitions.append(self.transitions(a.nextstate))
            elif a.ilabel == self.key_break:
                # transition to break: any break tokens
                for br, token in self.breaklist.items():
                    transitions.append(token)
            elif a.ilabel == self.key_space:
                # transition to space: any space tokens
                for sp, token in self.spacelist.items():
                    transitions.append(token)
            else:
                # transition to token: transition id is the token
                transitions.append(a.ilabel)
        return transitions
