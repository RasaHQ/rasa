import pytest
import numpy as np
from typing import List, Text
import logging

from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT


@pytest.mark.parametrize(
    "input_sequence_length, model_name, should_overflow",
    [(20, "bert", False), (1000, "bert", True), (1000, "xlnet", False)],
)
def test_sequence_length_overflow_train(
    input_sequence_length: int, model_name: Text, should_overflow: bool
):
    component = HFTransformersNLP({"model_name": model_name}, skip_model_load=True)
    message = Message.build(text=" ".join(["hi"] * input_sequence_length))
    if should_overflow:
        with pytest.raises(RuntimeError):
            component._validate_sequence_lengths(
                [input_sequence_length], [message], "text", inference_mode=False
            )
    else:
        component._validate_sequence_lengths(
            [input_sequence_length], [message], "text", inference_mode=False
        )


@pytest.mark.parametrize(
    "sequence_embeddings, actual_sequence_lengths, model_name, padding_needed",
    [
        (np.ones((1, 512, 5)), [1000], "bert", True),
        (np.ones((1, 512, 5)), [1000], "xlnet", False),
        (np.ones((1, 256, 5)), [256], "bert", False),
    ],
)
def test_long_sequences_extra_padding(
    sequence_embeddings: np.ndarray,
    actual_sequence_lengths: List[int],
    model_name: Text,
    padding_needed: bool,
):
    component = HFTransformersNLP({"model_name": model_name}, skip_model_load=True)
    modified_sequence_embeddings = component._add_extra_padding(
        sequence_embeddings, actual_sequence_lengths
    )
    if not padding_needed:
        assert np.all(modified_sequence_embeddings) == np.all(sequence_embeddings)
    else:
        assert modified_sequence_embeddings.shape[1] == actual_sequence_lengths[0]
        assert (
            modified_sequence_embeddings[0].shape[-1]
            == sequence_embeddings[0].shape[-1]
        )
        zero_embeddings = modified_sequence_embeddings[0][
            sequence_embeddings.shape[1] :
        ]
        assert np.all(zero_embeddings == 0)


@pytest.mark.parametrize(
    "token_ids, max_sequence_length_model, resulting_length, padding_added",
    [
        ([[1] * 200], 512, 512, True),
        ([[1] * 700], 512, 512, False),
        ([[1] * 200], 200, 200, False),
    ],
)
def test_input_padding(
    token_ids: List[List[int]],
    max_sequence_length_model: int,
    resulting_length: int,
    padding_added: bool,
):
    component = HFTransformersNLP(
        {"model_name": "bert", "model_weights": "bert-base-uncased"},
        skip_model_load=True,
    )
    component.pad_token_id = 0
    padded_input = component._add_padding_to_batch(token_ids, max_sequence_length_model)
    assert len(padded_input[0]) == resulting_length
    if padding_added:
        original_length = len(token_ids[0])
        assert np.all(np.array(padded_input[0][original_length:]) == 0)


@pytest.mark.parametrize(
    "sequence_length, model_name, should_overflow",
    [(1000, "bert", True), (256, "bert", False)],
)
def test_log_longer_sequence(
    sequence_length: int, model_name: Text, should_overflow: bool, caplog
):
    transformers_config = {
        "model_name": model_name,
        "model_weights": "bert-base-uncased",
    }

    transformers_nlp = HFTransformersNLP(transformers_config)

    text = " ".join(["hi"] * sequence_length)
    message = Message.build(text)

    caplog.set_level(logging.DEBUG)
    transformers_nlp.process(message)
    if should_overflow:
        assert "hi hi hi" in caplog.text
    assert message.get("text_language_model_doc") is not None


@pytest.mark.parametrize(
    "actual_sequence_length, max_input_sequence_length, zero_start_index",
    [(256, 512, 256), (700, 700, 700), (700, 512, 512)],
)
def test_attention_mask(
    actual_sequence_length: int, max_input_sequence_length: int, zero_start_index: int
):
    component = HFTransformersNLP(
        {"model_name": "bert", "model_weights": "bert-base-uncased"},
        skip_model_load=True,
    )

    attention_mask = component._compute_attention_mask(
        [actual_sequence_length], max_input_sequence_length
    )
    mask_ones = attention_mask[0][:zero_start_index]
    mask_zeros = attention_mask[0][zero_start_index:]

    assert np.all(mask_ones == 1)
    assert np.all(mask_zeros == 0)


# TODO: need to fix this failing test
@pytest.mark.skip(reason="Results in random crashing of github action workers")
@pytest.mark.parametrize(
    "model_name, model_weights, texts, expected_tokens, expected_indices",
    [
        (
            "bert",
            None,
            [
                "Good evening.",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["good", "evening"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sentence",
                    "i",
                    "want",
                    "em",
                    "bed",
                    "ding",
                    "s",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 12)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 20),
                    (21, 22),
                    (23, 27),
                    (28, 30),
                    (30, 33),
                    (33, 37),
                    (37, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "bert",
            "bert-base-chinese",
            [
                "晚上好",  # normal & easy case
                "没问题！",  # `！` is a Chinese punctuation
                "去东畈村",  # `畈` is a OOV token for bert-base-chinese
                "好的😃",  # include a emoji which is common in Chinese text-based chat
            ],
            [
                ["晚", "上", "好"],
                ["没", "问", "题", "！"],
                ["去", "东", "畈", "村"],
                ["好", "的", "😃"],
            ],
            [
                [(0, 1), (1, 2), (2, 3)],
                [(0, 1), (1, 2), (2, 3), (3, 4)],
                [(0, 1), (1, 2), (2, 3), (3, 4)],
                [(0, 1), (1, 2), (2, 3)],
            ],
        ),
        (
            "gpt",
            None,
            [
                "Good evening.",
                "hello",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["good", "evening"],
                ["hello"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                ["here", "is", "the", "sentence", "i", "want", "embe", "ddings", "for"],
            ],
            [
                [(0, 4), (5, 12)],
                [(0, 5)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 20),
                    (21, 22),
                    (23, 27),
                    (28, 32),
                    (32, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "gpt2",
            None,
            [
                "Good evening.",
                "hello",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["Good", "even", "ing"],
                ["hello"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sent",
                    "ence",
                    "I",
                    "want",
                    "embed",
                    "d",
                    "ings",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 9), (9, 12)],
                [(0, 5)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 16),
                    (16, 20),
                    (21, 22),
                    (23, 27),
                    (28, 33),
                    (33, 34),
                    (34, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "xlnet",
            None,
            [
                "Good evening.",
                "hello",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["Good", "evening"],
                ["hello"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sentence",
                    "I",
                    "want",
                    "embed",
                    "ding",
                    "s",
                    "for",
                ],
            ],
            [4, 3, 4, 5, 5, 12],
        ),
        (
            "distilbert",
            None,
            [
                "Good evening.",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["good", "evening"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sentence",
                    "i",
                    "want",
                    "em",
                    "bed",
                    "ding",
                    "s",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 12)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 20),
                    (21, 22),
                    (23, 27),
                    (28, 30),
                    (30, 33),
                    (33, 37),
                    (37, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "roberta",
            None,
            [
                "Good evening.",
                "hello",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["Good", "even", "ing"],
                ["hello"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sent",
                    "ence",
                    "I",
                    "want",
                    "embed",
                    "d",
                    "ings",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 9), (9, 12)],
                [(0, 5)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 16),
                    (16, 20),
                    (21, 22),
                    (23, 27),
                    (28, 33),
                    (33, 34),
                    (34, 38),
                    (39, 42),
                ],
            ],
        ),
    ],
)
@pytest.mark.skip_on_windows
def test_hf_transformer_edge_cases(
    model_name, model_weights, texts, expected_tokens, expected_indices
):

    if model_weights is None:
        model_weights_config = {}
    else:
        model_weights_config = {"model_weights": model_weights}
    transformers_config = {**{"model_name": model_name}, **model_weights_config}

    hf_transformer = HFTransformersNLP(transformers_config)
    whitespace_tokenizer = WhitespaceTokenizer()

    for text, gt_tokens, gt_indices in zip(texts, expected_tokens, expected_indices):

        message = Message.build(text=text)
        tokens = whitespace_tokenizer.tokenize(message, TEXT)
        message.set(TOKENS_NAMES[TEXT], tokens)
        hf_transformer.process(message)

        assert [t.text for t in tokens] == gt_tokens
        assert [t.start for t in tokens] == [i[0] for i in gt_indices]
        assert [t.end for t in tokens] == [i[1] for i in gt_indices]
