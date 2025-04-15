# algorithms/run_task_1.py
# Author: @ledondodo

from algorithms.algo_1 import (
    load_model,
    calculate_likelihoods,
    mask_special_tokens,
    beam_search,
    compare_logits_scores,
    compare_generations_sampling,
)


def run_task_1():
    """
    Run Task 1: Language Model Exploration
    - 1. Compute metrics for a generated sequence: likelihoods
    |- 1a. Compare likelihoods using logits and scores
    |- 1b. Compare generations using sampling and greedy decoding
    - 2. Mask special tokens during generation
    - 3. Beam search
    """
    print("\n## TASK 1: Explore LLM ##\n")

    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

    input = "Write a story about a data scientist."
    model, tokenizer = load_model(model_name)

    # Tasks
    calculate_likelihoods(model, tokenizer, input, 50, True)
    logits_processor = mask_special_tokens(model, tokenizer, input, 100)
    beam_search(model, tokenizer, input, 100, logits_processor, 2)

    # Sanity check
    compare_logits_scores(model, tokenizer, input, 50)
    compare_generations_sampling(model, tokenizer, input, 50)
