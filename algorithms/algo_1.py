# algorithms/algo_1.py
# Author: @ledondodo

import numpy as np
import torch
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    LogitsProcessor,
)


def load_model(checkpoint):
    """
    Load model and tokenizer from checkpoint

    Args:
        checkpoint (str): checkpoint name

    Returns:
        model: model
        tokenizer: tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return model, tokenizer


def tokenizer_config(tokenizer, input):
    """
    Configure tokenizer for input
    and return tokenized input

    Args:
        tokenizer
        input (str): input text

    Returns:
        input_tokens (torch.Tensor): tokenized input
    """
    # Transform input
    messages = [{"role": "user", "content": input}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    input_tokens = tokenizer.encode(input_text, return_tensors="pt")
    return input_tokens


def calculate_likelihoods(model, tokenizer, input, n_tokens, display=False):
    """
    Task 1.1: Likelihoods
    Compute different likelihood representations for a generated sequence

    Args:
        model: model
        tokenizer: tokenizer
        input (str): input text
        n_tokens (int): number of tokens to generate
        display (bool): display results

    Returns:
        likelihood: float, likelihood
        geometric_mean: float, geometric mean
        log_likelihood: float, log likelihood
        log_normalized_likelihood: float, log normalized
    """
    if display:
        print("\n## TASK 1.1: Likelihoods ##\n")

    # Generate sequence
    input_tokens = tokenizer_config(tokenizer, input)
    outputs = model.generate(
        input_tokens,
        max_new_tokens=n_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )
    if display:
        print("\nSequence:\n", tokenizer.decode(outputs["sequences"][0]))

    # Compute likelihoods
    likelihood = 1
    for logits, token in zip(outputs["logits"], outputs["sequences"][0][13:]):
        probs = torch.softmax(logits, dim=1)
        likelihood *= float(probs[0][token])
    geometric_mean = likelihood ** (1 / n_tokens)
    log_likelihood = math.log(likelihood)
    log_normalized_likelihood = log_likelihood / n_tokens

    if display:
        print("\nLikelihood:", likelihood)
        print("Geometric mean:", geometric_mean)
        print("Log likelihood:", log_likelihood)
        print("Log normalized likelihood:", log_normalized_likelihood, "\n")

    return likelihood, geometric_mean, log_likelihood, log_normalized_likelihood


class BlacklistTokensProcessor(LogitsProcessor):
    """
    Blacklist class for LogitsProcessor

    Attributes:
        blacklist_token_ids (list): blacklisted token ids
    """

    def __init__(self, blacklist_token_ids):
        """
        Initialize BlacklistTokensProcessor

        Args:
            blacklist_token_ids (list): blacklisted token ids
        """
        self.blacklist_token_ids = blacklist_token_ids

    def __call__(self, input_ids, scores):
        """
        Apply blacklist to scores
        Replace blacklisted tokens with -inf scores

        Args:
            input_ids (torch.Tensor): input ids
            scores (torch.Tensor): scores

        Returns:
            scores (torch.Tensor): scores
        """
        for token_id in self.blacklist_token_ids:
            scores[:, token_id] = -float("inf")
        return scores


def mask_special_tokens(model, tokenizer, input, n_tokens):
    """
    Task 1.2: Mask Special Tokens
    Mask special tokens during generation

    Args:
        model: model
        tokenizer: tokenizer
        input (str): input text
        n_tokens (int): number of tokens to generate

    Returns:
        logits_processor (BlacklistTokensProcessor): blacklist processor
    """
    print("\n## TASK 1.2: Mask Special Tokens ##")

    # Create a blacklist of tokens
    blacklist_tokens = [
        ".",
        "!",
        "?",
        ",",
        ";",
        ":",
        "*",
        '"',
        "-",
        "–",
        "(",
        ")",
        "âĢĵ",
        # "Ċ", # line break
    ]

    # Find every vocab token that contains the blacklisted tokens
    vocab_tokens = np.array(list(tokenizer.vocab.keys()))
    vocab_mask = np.zeros(len(vocab_tokens), dtype=bool)
    for black_token in blacklist_tokens:
        vocab_mask = vocab_mask | (np.char.find(vocab_tokens, black_token) >= 0)
    blacklist_tokens = vocab_tokens[vocab_mask]
    blacklist_token_ids = tokenizer.convert_tokens_to_ids(blacklist_tokens)

    # Create a processor to blacklist tokens
    logits_processor = LogitsProcessorList()
    logits_processor.append(BlacklistTokensProcessor(blacklist_token_ids))

    # Generate sequence (with BlacklistTokensProcessor)
    input_tokens = tokenizer_config(tokenizer, input)
    outputs = model.generate(
        input_tokens,
        max_new_tokens=n_tokens,
        return_dict_in_generate=True,
        output_logits=True,
        logits_processor=logits_processor,
    )

    output_text = tokenizer.decode(outputs["sequences"][0])
    print("\nSequence:\n", output_text, "\n")

    # Code to explore the generated tokens
    token_pos = 23
    token_id = outputs["sequences"][0][13 + token_pos].item()
    print("Display of the token:", tokenizer.decode(token_id))
    print("Token ID:", token_id)
    print("Token:", tokenizer.convert_ids_to_tokens(token_id))

    return logits_processor


def beam_search(model, tokenizer, input, n_tokens, logits_processor, nbeams):
    """
    Task 1.3: Beam Search
    Generate sequence with Beam Search

    Args:
        model: model
        tokenizer: tokenizer
        input (str): input text
        n_tokens (int): number of tokens to generate
        logits_processor (LogitsProcessor): logits processor
        nbeams (int): number of beams
    """
    print("\n\n## TASK 1.3: Beam Search ##\n")

    # Generate sequence (with Blacklist and Beam)
    input_tokens = tokenizer_config(tokenizer, input)
    outputs = model.generate(
        input_tokens,
        max_new_tokens=n_tokens,
        logits_processor=logits_processor,
        return_dict_in_generate=True,
        output_logits=True,
        num_beams=nbeams,
    )

    output_text = tokenizer.decode(outputs["sequences"][0])
    print("Sequence:", output_text, "\n")


def compare_logits_scores(model, tokenizer, input, n_tokens):
    """
    Sanity check for task 1.1:
    Compare likelihoods using logits and scores

    Args:
        model: model
        tokenizer: tokenizer
        input (str): input text
        n_tokens (int): number of tokens to generate
    """
    print("\n## TASK 1.1a: Compare Logits and Scores ##")

    # Generate sequence
    input_tokens = tokenizer_config(tokenizer, input)
    outputs = model.generate(
        input_tokens,
        max_new_tokens=n_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
        output_scores=True,
    )
    print("\nSequence:\n", tokenizer.decode(outputs["sequences"][0]))

    # Compute likelihoods with scores
    likelihood = 1
    print("\nStats with 'scores':")
    for logits in outputs["scores"]:
        probs = torch.softmax(logits, dim=1)
        likelihood *= float(torch.max(probs))
    geometric_mean = likelihood ** (1 / n_tokens)
    print("Likelihood:", likelihood)
    print("Log-likelihood:", math.log(likelihood))
    print("Geometric mean:", geometric_mean)

    # Compute likelihoods with logits
    likelihood = 1
    print("\nStats with 'logits':")
    for logits in outputs["logits"]:
        probs = torch.softmax(logits, dim=1)
        likelihood *= float(torch.max(probs))
    geometric_mean = likelihood ** (1 / n_tokens)
    print("Likelihood:", likelihood)
    print("Log-likelihood:", math.log(likelihood))
    print("Geometric mean:", geometric_mean, "\n")


def compare_generations_sampling(model, tokenizer, input, n_tokens):
    """
    Sanity check for task 1.1:
    Compare generation outputs with and without sampling
    Goal: decode the most likely token at each position in the sequence

    Args:
        model: model
        tokenizer: tokenizer
        input (str): input text
        n_tokens (int): number of tokens to generate
    """
    print("\n## TASK 1.1b: Compare Generations Sampling ##")

    # 1. Generate sequence, do_sample=True
    input_tokens = tokenizer_config(tokenizer, input)
    outputs = model.generate(
        input_tokens,
        max_new_tokens=n_tokens,
        do_sample=True,
        return_dict_in_generate=True,
        output_logits=True,
    )
    print(
        "\nSequence (do_sample=True):\n",
        tokenizer.decode(outputs["sequences"][0]),
        "\n",
    )

    # Compare generated sequence with ideal sequence
    print(
        "Generated sequence (do_sample=True):\n",
        outputs["sequences"][0][13:].tolist(),
        "\n",
    )
    ## Ideal sequence is the most likely token at each position (argmax of logits)
    ## outputs['logits'] is a list of 'n_tokens' tensors of shape (1, vocab_size)
    token_ids = [torch.argmax(logits).item() for logits in outputs["logits"]]
    print("Ideal sequence (do_sample=True):\n", token_ids, "\n")
    # Is there any different element?
    print(
        "Are the sequences the same? (do_sample=True)",
        bool(torch.all(torch.tensor(token_ids) == outputs["sequences"][0][13:])),
        "\n",
    )

    print("---")

    # 2. Generate sequence, do_sample=False
    outputs = model.generate(
        input_tokens,
        max_new_tokens=n_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )
    print(
        "\nSequence (do_sample=False):\n",
        tokenizer.decode(outputs["sequences"][0]),
        "\n",
    )

    # Compare generated sequence with ideal sequence
    print(
        "Generated sequence (do_sample=False):\n",
        outputs["sequences"][0][13:].tolist(),
        "\n",
    )
    token_ids = [torch.argmax(logits).item() for logits in outputs["logits"]]
    print("Ideal sequence (do_sample=False):\n", token_ids, "\n")
    print(
        "Are the sequences the same? (do_sample=False)",
        bool(torch.all(torch.tensor(token_ids) == outputs["sequences"][0][13:])),
        "\n",
    )
