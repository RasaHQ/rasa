import pytest
import numpy as np
from typing import List, Text
import logging

from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.shared.nlu.training_data.message import Message


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
@pytest.mark.skip_on_windows
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
