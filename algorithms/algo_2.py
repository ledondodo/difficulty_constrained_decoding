# algorithms/algo_2.py
# Author: @ledondodo

import torch
from algorithms.algo_1 import tokenizer_config


def dynamic_beam_search_naive(model, tokenizer, input, n_tokens, pmax, bmax):
    """
    Task 2: Dynamic Beam Search, pseudocode implementation (naive)

    Args:
        model: model
        tokenizer: tokenizer
        input (str): input text
        n_tokens (int): number of tokens to generate
        pmax (float): maximum probability to keep
        bmax (int): maximum number of beams to keep

    Returns:
        beam_best (tuple): best beam (input, probability, finished sequence)
    """
    # Configuration
    input_tokens = tokenizer_config(tokenizer, input)
    beams = [(input_tokens, 1, False)]  # (input, probability, finished sequence)
    print("Initial beam sequence:\n", tokenizer.decode(input_tokens[0]))

    # Dynamic Beam Search at each generation step
    for i in range(n_tokens):
        beams_next = []

        for b in beams:
            if b[2]:
                # If beam is finished, keep it
                beams_next.append(beams[b])
            else:
                # Generate next tokens for each beam
                output = model.generate(
                    b[0],
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_logits=True,
                )
                probs = torch.softmax(output["logits"][0], dim=1)

                # Create every possible beams, with next token probabilities (vectorized version)
                tokens = torch.arange(tokenizer.vocab_size).view(-1, 1)
                input_next = b[0].repeat(tokens.shape[0], 1)
                input_next = torch.cat([input_next, tokens], dim=-1).view(
                    tokens.shape[0], 1, -1
                )
                prob_next = b[1] * probs[0]
                finished_next = (tokens == tokenizer.eos_token_id).view(-1)
                beams_next.extend(list(zip(input_next, prob_next, finished_next)))

        # Sort next beams by probabilities
        beams_next = sorted(beams_next, key=lambda x: x[1], reverse=True)
        # Reset old beams
        beams = []
        # Keep the most probable beams until Pmax or Bmax
        p = 0
        while p < pmax and len(beams) < bmax:
            beams.append(beams_next[0])
            p += beams_next[0][1]
            beams_next = beams_next[1:]
        print("step", i + 1, "/", n_tokens, ", nbeams:", len(beams), ", p:", float(p))
        print(tokenizer.decode(beams[0][0][0]), "\n")

    # At the end, keep the most probable beam
    beam_best = max(beams, key=lambda x: x[1])
    output_text = tokenizer.decode(beam_best[0][0])
    print("\n** Final Beam **")
    print("p=", float(beam_best[1]), "\n", beam_best[0], "\n", output_text)

    return beam_best
