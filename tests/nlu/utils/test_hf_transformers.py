import pytest
import numpy as np

from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.nlu.training_data import Message


@pytest.mark.parametrize(
    "input_sequence_length, model_name, should_overflow",
    [(20, "bert", False), (1000, "bert", True), (1000, "xlnet", False)],
)
def test_sequence_length_overflow_train(
    input_sequence_length, model_name, should_overflow
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
    sequence_embeddings, actual_sequence_lengths, model_name, padding_needed
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
    token_ids, max_sequence_length_model, resulting_length, padding_added
):
    component = HFTransformersNLP({"model_name": "bert"}, skip_model_load=True)
    component.pad_token_id = 0
    padded_input = component._add_padding_to_batch(token_ids, max_sequence_length_model)
    assert len(padded_input[0]) == resulting_length
    if padding_added:
        original_length = len(token_ids[0])
        assert np.all(np.array(padded_input[0][original_length:]) == 0)
