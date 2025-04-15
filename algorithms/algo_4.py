# algorithms/algo_4.py
# Author: @ledondodo

import numpy as np
from algorithms.algo_1 import tokenizer_config
from transformers import AutoTokenizer


def extend_word_list(words):
    """
    Extend a canon list of words to include common variations

    Args:
        words (list): list of strings
    """
    extended_words = []
    for word in words:
        # Add original word
        extended_words.append(word)

        # Add word with leading/trailing spaces
        extended_words.append(f" {word}")
        extended_words.append(f"{word} ")
        extended_words.append(f" {word} ")

        # Add case variations (lowercase and uppercase)
        extended_words.append(word.lower())
        extended_words.append(word.upper())
        extended_words.append(word.title())

        # Optionally: add other variations if needed
        # (e.g., common misspellings, alternative representations)

    # Remove duplicates and return the extended list
    return list(dict.fromkeys(extended_words))


def words_mask(
    checkpoint,
    display,
):
    """
    Task 4: Words Mask
    Find possible next tokens to complete a sequence of tokens, given a list of words

    Args:
        checkpoint (str): model checkpoint name
        display (bool): display output
    """
    # Model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    words = [
        "atetitotut",
        "atatet",
        "etat",
        "otetit",
        # 'directions',
        # 'directors',
        # 'directiverse',
        # 'data',
        # 'unicorn',
        # 'universe',
        # 'ornament',
        # 'orn',
        # 'unable',
    ]
    # words = extend_word_list(words)
    seq = "i like atotet"

    words_tokens = np.array([tokenizer(w).input_ids for w in words], dtype=object)
    seq_tokens = tokenizer(seq).input_ids
    key = seq_tokens[-1]

    print("\n** Word Finder **")
    print(f"seq: '{seq}' {[tokenizer.decode(t) for t in seq_tokens]}")
    print(f"key: '{tokenizer.decode(key)}' ({key})")

    # get depth
    depth = 0
    for w in words_tokens:
        if len(w) > depth:
            depth = len(w)

    if display:
        print(f"\nWords: (depth {depth})")
        print([w for w in words])

    # match key in words tokens
    ## pad with False
    match_key = np.full((len(words), depth), False)
    for i, w in enumerate(words_tokens):
        for j, t in enumerate(w):
            if t == key:
                match_key[i, j] = True
    print(f"\nMatches (#word, #token): {np.where(match_key)}")

    # store results
    allowed_new = False
    allowed_tokens = []

    # find each key match
    for idx, idy in zip(*np.where(match_key)):
        if idy < len(seq_tokens):
            match_seq = seq_tokens[-(idy + 1) :] == words_tokens[idx][: idy + 1]
            print(
                f"> {match_seq}{' ' if match_seq else ''} "
                f"({idx},{idy}) '{words[idx]}'"
            )
            if match_seq:
                print(f"\t+ Match: ({idx},{idy}) '{words[idx]}'")
                # allow next token
                if idy < len(words_tokens[idx]) - 1:
                    allowed_tokens.append(words_tokens[idx][idy + 1])
                # end of a word, allow new words
                else:
                    allowed_new = True
    # if no match, allow new words
    if not allowed_tokens:
        print("\nNo match found... Allowing new words.\n")
        allowed_new = True

    # allow every word first token
    if allowed_new:
        for w in words_tokens:
            allowed_tokens.append(w[0])

    # remove duplicates
    allowed_tokens = list(set(allowed_tokens))

    if display:
        print(
            f"\nAllowed next tokens (sequence '{seq}', key: '{tokenizer.decode(key)}' {key}):"
        )
        print(f"> {[tokenizer.decode(t) for t in allowed_tokens]}")
