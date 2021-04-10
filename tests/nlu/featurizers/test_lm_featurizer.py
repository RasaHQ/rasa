import os
from typing import Text, List, Dict, Tuple

import numpy as np
import pytest
import logging

from _pytest.logging import LogCaptureFixture

from rasa.nlu.constants import (
    TOKENS_NAMES,
    NUMBER_OF_SUB_TOKENS,
    SEQUENCE_FEATURES,
    SENTENCE_FEATURES,
    LANGUAGE_MODEL_DOCS,
)
from rasa.nlu.tokenizers.lm_tokenizer import LanguageModelTokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.shared.nlu.constants import TEXT, INTENT


def skip_on_CI(model_name: Text, model_weights: Text) -> bool:
    """Checks whether to skip this configuration on CI.

    Only applies when skip_model_load=False
    """
    # First check if CI
    return (
        bool(os.environ.get("CI"))
        and model_name == "bert"
        and (not model_weights or model_weights == "rasa/LaBSE")
    )


def create_pretrained_transformers_config(
    model_name: Text, model_weights: Text
) -> Dict[Text, Text]:
    """Create a config for LanguageModelFeaturizer.

    If CI, skips model/model_weight combinations that are too large (bert with
    LaBSE).

    Args:
        model_name: model name
        model_weights: model weights name
    """
    if skip_on_CI(model_name, model_weights):
        pytest.skip(
            "Reason: this model is too large, loading it results in"
            "crashing of GH action workers."
        )
    config = {"model_name": model_name}
    if model_weights:
        config["model_weights"] = model_weights
    return config


def train_texts(
    texts: List[Text], model_name: Text, model_weights: Text
) -> List[Message]:
    config = create_pretrained_transformers_config(model_name, model_weights)
    whitespace_tokenizer = WhitespaceTokenizer()
    lm_featurizer = LanguageModelFeaturizer(config)

    messages = [Message.build(text=text) for text in texts]
    td = TrainingData(messages)

    whitespace_tokenizer.train(td)
    lm_featurizer.train(td)
    return messages


def process_texts(
    texts: List[Text], model_name: Text, model_weights: Text
) -> List[Message]:
    config = create_pretrained_transformers_config(model_name, model_weights)
    whitespace_tokenizer = WhitespaceTokenizer()
    lm_featurizer = LanguageModelFeaturizer(config)

    messages = []
    for text in texts:
        message = Message.build(text=text)
        whitespace_tokenizer.process(message)
        lm_featurizer.process(message)
        messages.append(message)
    return messages


@pytest.mark.parametrize(
    "model_name, model_weights, texts, expected_shape, expected_sequence_vec, expected_cls_vec",
    [
        (
            "bert",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [0.6569931, 0.77279466],
                [0.21718428, 0.34955627, 0.59124136, 0.6869872, 0.16993292],
            ],
            [
                [0.29528213, 0.5543281, -0.4091331, 0.65817744, 0.81740487],
                [-0.17215663, 0.26811457, -0.1922609, -0.63926417, -1.626383],
            ],
        ),
        (
            "bert",
            "bert-base-uncased",
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [0.57274431, -0.16078192],
                [-0.54851216, 0.09632845, -0.42788929, 0.11438307, 0.18316516],
            ],
            [
                [0.06880389, 0.32802248, -0.11250392, -0.11338016, -0.37116382],
                [0.05909365, 0.06433402, 0.08569094, -0.16530040, -0.11396892],
            ],
        ),
        (
            "gpt",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [-0.0630323737859726, 0.4029877185821533],
                [
                    0.8072432279586792,
                    -0.08990508317947388,
                    0.9985930919647217,
                    -0.38779014348983765,
                    0.08921952545642853,
                ],
            ],
            [
                [
                    0.16997766494750977,
                    0.1493849903345108,
                    0.39421725273132324,
                    -0.5753618478775024,
                    0.05096133053302765,
                ],
                [
                    0.41056010127067566,
                    -0.1169343888759613,
                    -0.3019704818725586,
                    -0.40207183361053467,
                    0.6289798021316528,
                ],
            ],
        ),
        (
            "gpt2",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [-0.03382749, -0.05373593],
                [-0.18434484, -0.5386464, -0.11122551, -0.95434338, 0.28311089],
            ],
            [
                [
                    -0.04710008203983307,
                    -0.2793063223361969,
                    -0.23804056644439697,
                    -0.3212292492389679,
                    0.11430201679468155,
                ],
                [
                    -0.1809544414281845,
                    -0.017152192071080208,
                    -0.3176477551460266,
                    -0.008387327194213867,
                    0.3365338146686554,
                ],
            ],
        ),
        (
            "xlnet",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [1.7612367868423462, 2.5819129943847656],
                [
                    0.784195065498352,
                    0.7068007588386536,
                    1.5883606672286987,
                    1.891886591911316,
                    2.5209126472473145,
                ],
            ],
            [
                [
                    2.171574831008911,
                    -1.5377449989318848,
                    -3.2671749591827393,
                    0.22520869970321655,
                    -1.598855972290039,
                ],
                [
                    1.6516317129135132,
                    0.021670114248991013,
                    -2.5114030838012695,
                    1.447351098060608,
                    -2.5866634845733643,
                ],
            ],
        ),
        (
            "distilbert",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [0.22866562008857727, -0.0575055330991745],
                [
                    -0.6448041796684265,
                    -0.5105321407318115,
                    -0.4892978072166443,
                    0.17531153559684753,
                    0.22717803716659546,
                ],
            ],
            [
                [
                    -0.09814466536045074,
                    -0.07325993478298187,
                    0.22358475625514984,
                    -0.20274735987186432,
                    -0.07363069802522659,
                ],
                [
                    -0.146609365940094,
                    -0.07373693585395813,
                    0.016850866377353668,
                    -0.2407529354095459,
                    -0.0979844480752945,
                ],
            ],
        ),
        (
            "roberta",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [-0.3092685, 0.09567838],
                [0.02152853, -0.08026707, -0.1080862, 0.12423468, -0.05378958],
            ],
            [
                [
                    -0.03930358216166496,
                    0.034788478165864944,
                    0.12246038764715195,
                    0.08401528000831604,
                    0.7026961445808411,
                ],
                [
                    -0.018586941063404083,
                    -0.09835464507341385,
                    0.03242188319563866,
                    0.09366855770349503,
                    0.4458026587963104,
                ],
            ],
        ),
    ],
)
class TestShapeValuesTrainAndProcess:
    @staticmethod
    def evaluate_message_shape_values(
        messages: List[Message],
        expected_shape: List[tuple],
        expected_sequence_vec: List[List[float]],
        expected_cls_vec: List[List[float]],
    ) -> None:
        for index in range(len(messages)):
            (computed_sequence_vec, computed_sentence_vec) = messages[
                index
            ].get_dense_features(TEXT, [])
            if computed_sequence_vec:
                computed_sequence_vec = computed_sequence_vec.features
            if computed_sentence_vec:
                computed_sentence_vec = computed_sentence_vec.features

            assert computed_sequence_vec.shape[0] == expected_shape[index][0] - 1
            assert computed_sequence_vec.shape[1] == expected_shape[index][1]
            assert computed_sentence_vec.shape[0] == 1
            assert computed_sentence_vec.shape[1] == expected_shape[index][1]

            # Look at the value of first dimension for a few starting timesteps
            assert np.allclose(
                computed_sequence_vec[: len(expected_sequence_vec[index]), 0],
                expected_sequence_vec[index],
                atol=1e-4,
            )

            # Look at the first value of first five dimensions
            assert np.allclose(
                computed_sentence_vec[0][:5], expected_cls_vec[index], atol=1e-4
            )

            (intent_sequence_vec, intent_sentence_vec) = messages[
                index
            ].get_dense_features(INTENT, [])
            if intent_sequence_vec:
                intent_sequence_vec = intent_sequence_vec.features
            if intent_sentence_vec:
                intent_sentence_vec = intent_sentence_vec.features

            assert intent_sequence_vec is None
            assert intent_sentence_vec is None

    def test_lm_featurizer_shape_values_train(
        self,
        model_name: Text,
        model_weights: Text,
        texts: List[Text],
        expected_shape: List[Tuple[int]],
        expected_sequence_vec: List[List[float]],
        expected_cls_vec: List[List[float]],
    ):
        messages = train_texts(texts, model_name, model_weights)
        self.evaluate_message_shape_values(
            messages, expected_shape, expected_sequence_vec, expected_cls_vec
        )

    def test_lm_featurizer_shape_values_process(
        self,
        model_name: Text,
        model_weights: Text,
        texts: List[Text],
        expected_shape: List[Tuple[int]],
        expected_sequence_vec: List[List[float]],
        expected_cls_vec: List[List[float]],
    ):
        messages = process_texts(texts, model_name, model_weights)
        self.evaluate_message_shape_values(
            messages, expected_shape, expected_sequence_vec, expected_cls_vec
        )


@pytest.mark.parametrize(
    "model_name, model_weights, texts, expected_number_of_sub_tokens",
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
                "sentence embeddings",
            ],
            [[1, 1], [1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1, 1, 1, 1, 2, 1], [1, 2]],
        ),
        (
            "bert",
            "bert-base-chinese",
            [
                "晚上好",  # normal & easy case
                "没问题！",  # `！` is a Chinese punctuation
                "去东畈村",  # `畈` is a OOV token for bert-base-chinese
                "好的😃",
                # include a emoji which is common in Chinese text-based chat
            ],
            [[3], [4], [4], [3]],
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
                "sentence embeddings",
            ],
            [
                [1, 1],
                [1],
                [1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1, 1, 1, 1, 2, 1],
                [1, 2],
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
                "sentence embeddings",
            ],
            [
                [1, 2],
                [1],
                [1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1, 2, 1, 1, 3, 1],
                [2, 3],
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
                "sentence embeddings",
            ],
            [
                [1, 1],
                [1],
                [1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1, 1, 1, 1, 3, 1],
                [1, 3],
            ],
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
                "sentence embeddings",
            ],
            [[1, 1], [1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1, 1, 1, 1, 4, 1], [1, 4]],
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
                "sentence embeddings",
            ],
            [
                [1, 2],
                [1],
                [1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1, 2, 1, 1, 3, 1],
                [2, 3],
            ],
        ),
        (
            "bert",
            "bert-base-uncased",
            [
                "Good evening.",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
                "sentence embeddings",
            ],
            [[1, 1], [1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1, 1, 1, 1, 4, 1], [1, 4]],
        ),
    ],
)
class TestSubTokensTrainAndProcess:
    @staticmethod
    def check_subtokens(
        texts: List[Text],
        messages: List[Message],
        expected_number_of_sub_tokens: List[List[float]],
    ):
        whitespace_tokenizer = WhitespaceTokenizer()
        for index, message in enumerate(messages):
            assert [
                t.get(NUMBER_OF_SUB_TOKENS) for t in message.get(TOKENS_NAMES[TEXT])
            ] == expected_number_of_sub_tokens[index]
            assert len(message.get(TOKENS_NAMES[TEXT])) == len(
                whitespace_tokenizer.tokenize(Message.build(text=texts[index]), TEXT)
            )

    def test_lm_featurizer_number_of_sub_tokens_train(
        self,
        model_name: Text,
        model_weights: Text,
        texts: List[Text],
        expected_number_of_sub_tokens: List[List[float]],
    ):
        messages = train_texts(texts, model_name, model_weights)
        self.check_subtokens(texts, messages, expected_number_of_sub_tokens)

    def test_lm_featurizer_number_of_sub_tokens_process(
        self,
        model_name: Text,
        model_weights: Text,
        texts: List[Text],
        expected_number_of_sub_tokens: List[List[float]],
    ):
        messages = process_texts(texts, model_name, model_weights)
        self.check_subtokens(texts, messages, expected_number_of_sub_tokens)


@pytest.mark.parametrize(
    "input_sequence_length, model_name, should_overflow",
    [(20, "bert", False), (1000, "bert", True), (1000, "xlnet", False)],
)
def test_sequence_length_overflow_train(
    input_sequence_length: int, model_name: Text, should_overflow: bool
):
    component = LanguageModelFeaturizer(
        {"model_name": model_name}, skip_model_load=True
    )
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
    component = LanguageModelFeaturizer(
        {"model_name": model_name}, skip_model_load=True
    )
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
    component = LanguageModelFeaturizer({"model_name": "bert"}, skip_model_load=True)
    component.pad_token_id = 0
    padded_input = component._add_padding_to_batch(token_ids, max_sequence_length_model)
    assert len(padded_input[0]) == resulting_length
    if padding_added:
        original_length = len(token_ids[0])
        assert np.all(np.array(padded_input[0][original_length:]) == 0)


@pytest.mark.parametrize(
    "sequence_length, model_name, model_weights, should_overflow",
    [
        (1000, "bert", "bert-base-uncased", True),
        (256, "bert", "bert-base-uncased", False),
    ],
)
def test_log_longer_sequence(
    sequence_length: int,
    model_name: Text,
    model_weights: Text,
    should_overflow: bool,
    caplog: LogCaptureFixture,
):
    config = {"model_name": model_name, "model_weights": model_weights}

    featurizer = LanguageModelFeaturizer(config)

    text = " ".join(["hi"] * sequence_length)
    tokenizer = WhitespaceTokenizer()
    message = Message.build(text=text)
    td = TrainingData([message])
    tokenizer.train(td)
    caplog.set_level(logging.DEBUG)
    featurizer.process(message)
    if should_overflow:
        assert "hi hi hi" in caplog.text
    assert len(message.features) >= 2


@pytest.mark.parametrize(
    "actual_sequence_length, max_input_sequence_length, zero_start_index",
    [(256, 512, 256), (700, 700, 700), (700, 512, 512)],
)
def test_attention_mask(
    actual_sequence_length: int, max_input_sequence_length: int, zero_start_index: int
):
    component = LanguageModelFeaturizer({"model_name": "bert"}, skip_model_load=True)

    attention_mask = component._compute_attention_mask(
        [actual_sequence_length], max_input_sequence_length
    )
    mask_ones = attention_mask[0][:zero_start_index]
    mask_zeros = attention_mask[0][zero_start_index:]

    assert np.all(mask_ones == 1)
    assert np.all(mask_zeros == 0)


def test_log_deprecation_warning_with_old_config(caplog: LogCaptureFixture):
    message = Message.build("hi there")

    transformers_nlp = HFTransformersNLP(
        {"model_name": "bert", "model_weights": "bert-base-uncased"}
    )
    transformers_nlp.process(message)

    caplog.set_level(logging.DEBUG)
    lm_tokenizer = LanguageModelTokenizer()
    lm_tokenizer.process(message)
    lm_featurizer = LanguageModelFeaturizer(skip_model_load=True)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        lm_featurizer.process(message)

    assert "deprecated component HFTransformersNLP" in caplog.text


def test_preserve_sentence_and_sequence_features_old_config():
    attribute = TEXT
    message = Message.build("hi there")

    transformers_nlp = HFTransformersNLP(
        {"model_name": "bert", "model_weights": "bert-base-uncased"}
    )
    transformers_nlp.process(message)
    lm_tokenizer = LanguageModelTokenizer()
    lm_tokenizer.process(message)

    lm_featurizer = LanguageModelFeaturizer({"model_name": "gpt2"})
    lm_featurizer.process(message)

    message.set(LANGUAGE_MODEL_DOCS[attribute], None)
    lm_docs = lm_featurizer._get_docs_for_batch(
        [message], attribute=attribute, inference_mode=True
    )[0]
    hf_docs = transformers_nlp._get_docs_for_batch(
        [message], attribute=attribute, inference_mode=True
    )[0]
    assert not (message.features[0].features == lm_docs[SEQUENCE_FEATURES]).any()
    assert not (message.features[1].features == lm_docs[SENTENCE_FEATURES]).any()
    assert (message.features[0].features == hf_docs[SEQUENCE_FEATURES]).all()
    assert (message.features[1].features == hf_docs[SENTENCE_FEATURES]).all()


@pytest.mark.parametrize(
    "text, tokens, expected_feature_tokens",
    [
        (
            "购买 iPhone 12",  # whitespace ' ' is expected to be removed
            [("购买", 0), (" ", 2), ("iPhone", 3), (" ", 9), ("12", 10)],
            [("购买", 0), ("iPhone", 3), ("12", 10)],
        )
    ],
)
def test_lm_featurizer_correctly_handle_whitespace_token(
    text: Text,
    tokens: List[Tuple[Text, int]],
    expected_feature_tokens: List[Tuple[Text, int]],
):
    from rasa.nlu.tokenizers.tokenizer import Token

    config = {
        "model_name": "bert",
        "model_weights": "bert-base-chinese",
    }

    lm_featurizer = LanguageModelFeaturizer(config)

    message = Message.build(text=text)
    message.set(TOKENS_NAMES[TEXT], [Token(word, start) for (word, start) in tokens])

    result, _ = lm_featurizer._tokenize_example(message, TEXT)

    assert [(token.text, token.start) for token in result] == expected_feature_tokens


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
    lm_featurizer = LanguageModelFeaturizer(
        {"model_name": "bert"}, skip_model_load=True
    )
    lm_featurizer.pad_token_id = 0
    padded_input = lm_featurizer._add_padding_to_batch(
        token_ids, max_sequence_length_model
    )
    assert len(padded_input[0]) == resulting_length
    if padding_added:
        original_length = len(token_ids[0])
        assert np.all(np.array(padded_input[0][original_length:]) == 0)
