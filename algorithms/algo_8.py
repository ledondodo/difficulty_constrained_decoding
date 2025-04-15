# algorithms/algo_8.py
# Author: @ledondodo

import os
import torch
from src.utils import tokenizer_config
from transformers import (
    GenerationConfig,
    GenerationMixin,
    AutoTokenizer,
    LlamaForCausalLM,
    BeamScorer,
    LogitsProcessorList,
    LogitsProcessor,
    StoppingCriteriaList,
)
from typing import Optional, Union
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    _split_model_inputs,
    stack_model_outputs,
)
from tqdm import tqdm


GenerateBeamOutput = Union[
    GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput
]


class WhiteTokensProcessor(LogitsProcessor):
    """White tokens class for LogitsProcessor"""

    def __init__(self, white_token_ids):
        self.white_token_ids = white_token_ids

    def __call__(self, input_ids, scores):
        # Only keep scores or white tokens
        white_scores = torch.full_like(scores, -float("inf"))
        for token_id in self.white_token_ids:
            white_scores[:, token_id] = scores[:, token_id]
        return white_scores


class CustomGenerationConfig(GenerationConfig):
    def __init__(
        self,
        params,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.params = params


class ConstrainedGenerationMixin(GenerationMixin):
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
        wsm = generation_config.params["wsm"]
        display = generation_config.params["display"]
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

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

        print("\n** Generation **")
        pbar = tqdm(total=generation_config.max_new_tokens, desc="Generation")

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
            # configure whitelist processor
            processors = [LogitsProcessorList()] * num_beams
            for i, p in enumerate(processors):
                state = wsm.match(input_ids[i])
                whitelist = wsm.transitions(state)
                p.append(WhiteTokensProcessor(whitelist))

            # CUSTOM
            # apply processors (i.e. whitelists) according to counters values
            next_token_scores = torch.zeros_like(next_token_logits)
            for beam in range(num_beams):
                next_token_scores[beam] = processors[beam](
                    input_ids[[beam], :], next_token_logits[[beam], :]
                )

            # log softmax on the masked scores
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
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            # Beam token selection
            topk_scores, next_tokens = torch.topk(
                next_token_scores,
                num_beams,
                dim=1,
                largest=True,
                sorted=True,
            )

            if do_sample:
                # Sample from the top-k results
                probabilities = torch.softmax(topk_scores, dim=1)
                sampled_indices = torch.multinomial(
                    probabilities, num_samples=num_beams, replacement=False
                )
                # Re index
                topk_scores = topk_scores.gather(1, sampled_indices)
                next_tokens = next_tokens.gather(1, sampled_indices)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # CUSTOM
            beam_next_tokens = next_tokens[0]
            beam_idx = next_indices[0]
            input_ids = input_ids[beam_idx, :]
            beam_scores = next_beam_scores[beam_idx, beam_next_tokens]

            input_ids = torch.cat([input_ids, beam_next_tokens.unsqueeze(-1)], dim=-1)

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

            pbar.update(1)

        # CUSTOM
        ## remove beam_scorer.finalize
        sequence_outputs = {
            "sequences": input_ids[:num_return_sequences, :],
            "sequence_scores": topk_scores[0, :num_return_sequences],
            "beam_indices": next_indices[0, :num_return_sequences],
        }

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


class ConstrainedDecoding(ConstrainedGenerationMixin, LlamaForCausalLM):
    pass


def constrained_decoding(checkpoint, input_str, params):
    """
    Task 8: Constrained Decoding
    Generate with constrained decoding from the Pynini implementation

    Args:
        checkpoint (str): path to the model checkpoint
        input_str (str): input string
        params (dict): custom parameters
    """
    print("** Config for generation **")
    for key, value in params.items():
        print(f"{key} (default: {value})")
    print()

    # Model
    model = ConstrainedDecoding.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer_config(tokenizer, params["device"], input_str)

    # Configuration
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    params["tokenizer_id_2_str"] = tokenizer.convert_ids_to_tokens
    config = CustomGenerationConfig(
        params=params,  # custom
        max_new_tokens=params["N"],
        return_dict_in_generate=True,
        output_logits=True,
        num_beams=10,
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
    print("** Output tokens **")
    for i, token_id in enumerate(output.sequences[0][len(inputs[0]) :]):
        print(f"{tokenizer.decode(token_id)}", end="/")
    print("\n\n** Sequence Scores **\n", output.sequences[0], "\n")
