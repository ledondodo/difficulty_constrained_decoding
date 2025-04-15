# src/data.py
# Author: @ledondodo

import os
import pandas as pd


def filter_level(df, level):
    """Filter words from a DataFrame by CEFR level
    input: DataFrame, level
    output: set of unique words"""
    flt = df[df.cefr.apply(lambda x: level in x)]
    words = set(flt.word)
    print(
        f"Filter {level} words: {df.shape[0]} -> {flt.shape[0]} = {100*round(flt.shape[0]/df.shape[0],4):.2f}%",
        f"\n(unique: {len(set(df.word))} -> {len(words)} = {100*round(len(words)/len(set(df.word)),4):.2f}%)",
    )
    return words


class Data:
    def __init__(
        self, tokenizer, words=None, breaks=None, spaces=None, exp=True, size_limit=None
    ):
        print("\n** Loading Data... **")
        self.words = set() if words is None else words
        self.breaks = {".", ",", "!", "?", "-", ":", ";"} if breaks is None else breaks
        self.spaces = {" ", "\n", "\t"} if spaces is None else spaces
        self.whitelist = {}
        self.words_whitelist = {}
        self.breaklist = {}
        self.spacelist = {}
        self.vocab_size = tokenizer.vocab_size
        if exp:
            self.expand()
        if size_limit:
            self.words = set(list(self.words)[:size_limit])
        self.tokenize_data(tokenizer)

    @classmethod
    def from_csv(
        cls, tokenizer, path, level="A1", breaks=None, spaces=None, size_limit=None
    ):
        """Load Data from a CSV file
        input: path, level, breaks, spaces
        output: Data"""
        print("\n** Reading from CSV file... **")
        df = pd.read_csv(path)
        if path == "data/Kaggle.csv":
            # rename column headword to word and CEFR to cefr
            df = df.rename(columns={"headword": "word", "CEFR": "cefr"})
        words = filter_level(df, level)
        data = cls(
            tokenizer, words=words, breaks=breaks, spaces=spaces, size_limit=size_limit
        )
        return data

    def expand(self):
        """Expand data words with common variations"""
        words_exp = set()
        for w in self.words:
            words_exp.add(w)
            # Case variations
            words_exp.add(w.lower())
            words_exp.add(w.title())
            words_exp.add(w.upper())
        for w in words_exp.copy():
            # Special characters variations
            words_exp.add(" " + w)
        print(f"Expand words: {len(self.words)} -> {len(words_exp)}")
        self.words = words_exp

    def tokenize_data(self, tokenizer):
        """Tokenize the data into whitelist, words_whitelist, breaklist, spacelist
        input: tokenizer"""
        # Tokens from the tokenizer
        token_ids = list(tokenizer.get_vocab().values())
        for t_id in token_ids:
            t_str = tokenizer.decode([t_id])
            # Compare at a character level (str) tokens with data
            for word in self.words:
                if t_str in word:
                    self.whitelist[t_id] = t_str
                    self.words_whitelist.setdefault(word, {}).update({t_id: t_str})
            for br in self.breaks:
                if t_str == br:
                    self.breaklist[t_id] = t_str
            for sp in self.spaces:
                if t_str == sp:
                    self.spacelist[t_id] = t_str
        # Sort for reproducibility
        for word in self.words:
            self.words_whitelist[word] = dict(
                sorted(self.words_whitelist[word].items(), key=lambda item: item[0])
            )
        self.whitelist = dict(sorted(self.whitelist.items(), key=lambda item: item[0]))
        self.breaklist = dict(sorted(self.breaklist.items(), key=lambda item: item[0]))
        self.spacelist = dict(sorted(self.spacelist.items(), key=lambda item: item[0]))
