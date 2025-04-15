# algorithms/algo_3.py
# Author: @ledondodo

import os
import numpy as np
import torch
from algorithms.algo_1 import (
    tokenizer_config,
    BlacklistTokensProcessor,
)
from transformers import (
    GenerationConfig,
    GenerationMixin,
    AutoTokenizer,
    LlamaForCausalLM,
    BeamScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from typing import Optional, Union
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    _split_model_inputs,
    stack_model_outputs,
)
import matplotlib.pyplot as plt


GenerateBeamOutput = Union[
    GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput
]


def make_plot(
    data,
    dir,
    name,
    text,
    threshold=None,
    vline=None,
    log=False,
    scale=None,
    boxloc=None,
):
    """
    Make plot

    Args:
        data (torch.Tensor): data
        dir (str): save path
        name (str): file name
        text (dict): figure text data
        threshold (list): threshold
        vline (list): vertical lines for punctuation tokens
        log (bool): use logarithmic scale
        scale (int): scale
        boxloc (str): boxloc for figure legend
    """
    # Hyperparameters info
    hyperparameters = text["subtitle"]

    # Create directory
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.ion()
    plt.plot(data)
    if threshold:
        plt.plot(
            threshold,
            color="orange",
            linestyle="--",
            alpha=0.8,
            linewidth=0.8,
            label=f"threshold",
        )
    if hyperparameters["strategy"] == "A":
        threshold_text = f't0={hyperparameters["t0"]}, t1={hyperparameters["t1"]}, '
    else:
        threshold_text = f'pmax={hyperparameters["pmax"]}, '
    if vline:
        if hyperparameters["max_consecutive"] <= hyperparameters["N"]:
            for v in vline:
                plt.axvline(
                    x=v,
                    color="green",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=0.8,
                    label=f"punctuation token at N={v}",
                )
    plt.title(text["title"], fontsize=16, fontweight="bold", y=1.05)
    plt.figtext(
        0.5,
        0.9,
        f'strategy={hyperparameters["strategy"]}, '
        + f'alpha={hyperparameters["alpha"]}, '
        + threshold_text
        + f'bmax={hyperparameters["bmax"]}, '
        + f'blacklist={hyperparameters["use_blacklist"]}, '
        + f'max_consecutive={hyperparameters["max_consecutive"]}',
        ha="center",
        fontsize=10,
        style="italic",
        color="gray",
    )
    if boxloc:
        plt.legend(loc=boxloc)
    else:
        plt.legend(loc="upper right")
    plt.xlabel(text["xlabel"])
    plt.ylabel(text["ylabel"])
    plt.xticks(range(0, len(data), max(1, len(data) // 10)))
    if log:
        plt.yscale("log")
    elif scale:
        plt.yticks(np.arange(0, scale, scale // 20 + 1))
    else:
        plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig(f"{dir}/{name}.png")
    plt.show()
    input("Press Enter to close the plot window...")
    plt.close()


class CustomGenerationConfig(GenerationConfig):
    """
    Custom class for GenerationConfig

    Args:
        hyperparameters (dict): custom hyperparameters
    """

    def __init__(
        self,
        hyperparameters,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hyperparameters = hyperparameters


class CustomGenerationMixin(GenerationMixin):
    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: CustomGenerationConfig,
        synced_gpus: bool,
        logits_warper: Optional[LogitsProcessorList],
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample
        num_return_sequences = generation_config.num_return_sequences
        pmax = generation_config.hyperparameters["pmax"]  # custom
        bmax = generation_config.hyperparameters["bmax"]  # custom
        display = generation_config.hyperparameters["display"]  # custom
        blacklist = generation_config.hyperparameters["blacklist"]  # custom
        counter_max = generation_config.hyperparameters["max_consecutive"]  # custom
        t0 = generation_config.hyperparameters["t0"]  # custom
        t1 = generation_config.hyperparameters["t1"]  # custom
        alpha = generation_config.hyperparameters["alpha"]  # custom
        counters = torch.zeros(input_ids.shape[0])  # custom
        strategy = generation_config.hyperparameters["strategy"]  # custom
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )
        if display:
            print(
                f"\n** Custom beam search **\nstrategy={strategy}, pmax={pmax}, bmax={bmax}, blacklist={blacklist!=None}, max_consecutive={counter_max}\n"
            )
            print("N  | nbeams | prob | prob seq | counters (first 5)")

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # CUSTOM
        # configure blacklist processor
        blacklist_processor = LogitsProcessorList()
        if blacklist is not None:
            blacklist_processor.append(BlacklistTokensProcessor(blacklist))

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size))
            if (return_dict_in_generate and output_scores)
            else None
        )
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        # CUSTOM: plots
        data_beam_scores = torch.empty(input_ids.shape[0], 0)
        data_token_scores = []
        data_nbeams = []
        data_counter_reset = np.empty(input_ids.shape[0], dtype=object)
        for i in range(input_ids.shape[0]):
            data_counter_reset[i] = []
        data_threshold = []

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(**inputs_per_sub_batch, return_dict=True)
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)

            else:  # Unchanged original behavior
                outputs = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # CUSTOM
            # apply processors (i.e. blacklists) according to counters values
            next_token_scores = torch.zeros_like(next_token_logits)
            for beam in range(num_beams):
                if counters[beam] > counter_max:
                    next_token_scores[beam] = logits_processor(
                        input_ids[[beam], :], next_token_logits[[beam], :]
                    )
                else:
                    next_token_scores[beam] = blacklist_processor(
                        input_ids[[beam], :], next_token_logits[[beam], :]
                    )
            next_token_scores = torch.nn.functional.log_softmax(
                next_token_scores, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)
            next_beam_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_beam_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # CUSTOM
            N = input_ids.shape[1] - decoder_prompt_len + 1
            # prepare probs for topk: flatten, normalize by sequence length, then take exp
            vocab_size = next_beam_scores.shape[-1]
            # 1. flatten the scores
            flat_next_beam_scores = next_beam_scores.view(
                batch_size, num_beams * vocab_size
            )
            # 2. normalize probs
            if strategy == "D1":
                # softmax on flatten scores
                next_beam_probs = torch.softmax(
                    flat_next_beam_scores / N**alpha, dim=-1
                )
            elif strategy == "D2":
                # softmax on scores, then flatten
                next_beam_probs = torch.softmax(next_beam_scores / N**alpha, dim=-1)
                next_beam_probs = next_beam_probs.view(
                    batch_size, num_beams * vocab_size
                )
            else:
                next_beam_probs = torch.exp(flat_next_beam_scores / N**alpha)

            # Beam token selection
            if do_sample:
                next_tokens = torch.multinomial(next_beam_probs, num_samples=bmax)
                next_beam_probs = torch.gather(next_beam_probs, -1, next_tokens)
                next_beam_probs, _indices = torch.sort(
                    next_beam_probs, descending=True, dim=1
                )
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                topk_probs, next_tokens = torch.topk(
                    next_beam_probs,
                    bmax,
                    dim=1,
                    largest=True,
                    sorted=True,
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # CUSTOM
            # compute k (different strategies)
            if strategy == "A":
                if N == 1:
                    threshold = t0
                else:
                    threshold = threshold * t1
            elif strategy == "B":
                prob_mass = torch.sum(next_beam_probs, dim=-1)
                threshold = pmax * prob_mass
            elif strategy in ["C", "D1", "D2"]:
                threshold = pmax
            else:
                raise ValueError("Invalid strategy:", strategy)
            data_threshold.append(threshold)
            topk_cumsum = torch.cumsum(topk_probs, dim=-1)
            topk_mask = topk_cumsum < threshold
            dynamic_k = int(torch.sum(topk_mask, dim=-1)) + 1
            dynamic_k = max(1, min(dynamic_k, bmax))

            # CUSTOM
            # apply k to hyperparameters, and tensors shape
            num_beams = dynamic_k
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"][
                :dynamic_k, :
            ]
            ## remove beam_scorer process
            beam_next_tokens = next_tokens[0][:dynamic_k]
            beam_idx = next_indices[0][:dynamic_k]
            input_ids = input_ids[beam_idx, :]
            beam_scores = next_beam_scores[beam_idx, beam_next_tokens]
            data_beam_scores = data_beam_scores[beam_idx]

            # CUSTOM
            # control counters
            is_next_punct = (~np.isin(beam_next_tokens, blacklist)).astype(int)
            counters = (counters[beam_idx] + 1) * is_next_punct
            data_counter_reset = data_counter_reset[beam_idx.tolist()]
            for i in range(len(is_next_punct)):
                if is_next_punct[i] == 0:
                    data_counter_reset[i] = data_counter_reset[i] + [
                        input_ids.shape[1] + 1 - decoder_prompt_len
                    ]

            input_ids = torch.cat([input_ids, beam_next_tokens.unsqueeze(-1)], dim=-1)
            if display:
                print(
                    f"{input_ids.shape[1]-decoder_prompt_len}{(dynamic_k//4)*' '} | {dynamic_k} | {round(topk_cumsum[0,dynamic_k-1].item(),3)} | {round(torch.exp(beam_scores[0]).item(),3)} | {counters.tolist()[:5]}"
                )
            data_beam_scores = torch.cat(
                (data_beam_scores, torch.exp(beam_scores).unsqueeze(1)), dim=1
            )
            data_token_scores.append(topk_cumsum[0, dynamic_k - 1].item())
            data_nbeams.append(dynamic_k)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    (
                        beam_indices[beam_idx[i]] + (beam_idx[i],)
                        for i in range(len(beam_indices))
                    )
                )

            # increase cur_len
            cur_len = cur_len + 1

            if all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        # CUSTOM
        ## remove beam_scorer.finalize
        sequence_outputs = {
            "sequences": input_ids[:num_return_sequences, :],
            "sequence_scores": topk_probs[0, :num_return_sequences],
            "beam_indices": next_indices[0, :num_return_sequences],
        }

        # CUSTOM: plot data
        ## save in ../out/algo_3/
        print("\n\n** Plotting **\n")
        make_plot(
            data=data_beam_scores[0],
            dir="./out/algo_3",
            name="beam_prob",
            text={
                "title": "Top Beam Prob",
                "subtitle": generation_config.hyperparameters,
                "xlabel": "Sequence length",
                "ylabel": "Probability",
            },
            vline=data_counter_reset[0],
            boxloc="upper right",
        )
        make_plot(
            data=data_beam_scores[0],
            dir="./out/algo_3",
            name="beam_log",
            text={
                "title": "Top Beam Log Prob",
                "subtitle": generation_config.hyperparameters,
                "xlabel": "Sequence length",
                "ylabel": "Log Probability",
            },
            vline=data_counter_reset[0],
            log=True,
            boxloc="upper right",
        )
        make_plot(
            data=data_nbeams,
            dir="./out/algo_3",
            name="nbeams",
            text={
                "title": "Number of Beams",
                "subtitle": generation_config.hyperparameters,
                "xlabel": "Sequence length",
                "ylabel": "Number of Beams",
            },
            vline=data_counter_reset[0],
            scale=5 * (max(data_nbeams) // 5 + 1) + 1,
            boxloc="upper left",
        )
        make_plot(
            data=data_token_scores,
            dir="./out/algo_3",
            name="topk_cum_prob",
            text={
                "title": "Topk Cummulative Prob",
                "subtitle": generation_config.hyperparameters,
                "xlabel": "Sequence length",
                "ylabel": "Cumulative Probability",
            },
            threshold=data_threshold,
            vline=data_counter_reset[0],
            boxloc="upper right",
        )
        make_plot(
            data=data_token_scores,
            dir="./out/algo_3",
            name="topk_cum_log",
            text={
                "title": "Topk Cummulative Log Prob",
                "subtitle": generation_config.hyperparameters,
                "xlabel": "Sequence length",
                "ylabel": "Cumulative Probability",
            },
            threshold=data_threshold,
            vline=data_counter_reset[0],
            boxloc="upper right",
            log=True,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]


class CustomModel(CustomGenerationMixin, LlamaForCausalLM):
    """Custom model class"""

    pass


def get_blacklist(tokenizer):
    """
    Get blacklist of tokens

    Args:
        tokenizer: tokenizer

    Returns:
        blacklist (list): blacklist tokens ids
    """
    # Initialize black tokens
    ## add "Ċ" to blacklist line break
    black_tokens = [".", "!", "?", ",", ";", ":", "*", '"', "-", "–", "(", ")", "âĢĵ"]

    # Mask vocab tokens containing blacklisted tokens
    vocab_tokens = np.array(list(tokenizer.vocab.keys()))
    vocab_mask = np.zeros(len(vocab_tokens), dtype=bool)
    for black_token in black_tokens:
        vocab_mask = vocab_mask | (np.char.find(vocab_tokens, black_token) >= 0)

    # Extend blacklist
    black_tokens = vocab_tokens[vocab_mask]
    blacklist = tokenizer.convert_tokens_to_ids(black_tokens)

    return blacklist


def dynamic_beam_search(
    model_name,
    input_str,
    hyperparameters,
):
    """
    Task 3: Dynamic Beam Search

    Args:
        model_name (str): model name
        input_str (str): input string
        hyperparameters (dict): hyperparameters

    Returns:
        output (transformers.generation.utils.GenerateBeamDecoderOnlyOutput)
    """
    # Model
    model = CustomModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer_config(tokenizer, input_str)

    # Blacklist
    if hyperparameters["use_blacklist"]:
        blacklist = get_blacklist(tokenizer)
        print("Size of the blacklist", len(blacklist), "\n")
    else:
        blacklist = None
    hyperparameters["blacklist"] = blacklist

    # Configuration
    config = CustomGenerationConfig(
        hyperparameters=hyperparameters,  # custom
        max_new_tokens=hyperparameters["N"],
        return_dict_in_generate=True,
        output_logits=True,
        num_beams=2,  # >1 to use beam search
    )

    # Generate sequence
    output = model.generate(
        inputs,
        generation_config=config,
    )
    print("\n** Input **\n", tokenizer.decode(inputs[0]), "\n")
    print(
        "** Output **\n", tokenizer.decode(output.sequences[0][len(inputs[0]) :]), "\n"
    )

    return output
