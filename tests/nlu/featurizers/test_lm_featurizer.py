import os
from typing import Text, List, Dict, Tuple, Any, Callable

import numpy as np
import pytest
import logging

from _pytest.monkeypatch import MonkeyPatch
from _pytest.logging import LogCaptureFixture

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.nlu.constants import TOKENS_NAMES, NUMBER_OF_SUB_TOKENS
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.nlu.tokenizers.tokenizer import Token


@pytest.fixture
def resource_language_model_featurizer() -> Resource:
    return Resource("LanguageModelFeaturizer")


@pytest.fixture
def create_language_model_featurizer(
    default_model_storage: ModelStorage,
    resource_language_model_featurizer,
    default_execution_context: ExecutionContext,
) -> Callable[[Dict[Text, Any]], LanguageModelFeaturizer]:
    def inner(config: Dict[Text, Any]) -> LanguageModelFeaturizer:
        return LanguageModelFeaturizer.create(
            config={**LanguageModelFeaturizer.get_default_config(), **config},
            model_storage=default_model_storage,
            resource=resource_language_model_featurizer,
            execution_context=default_execution_context,
        )

    return inner


def skip_on_CI_with_bert(model_name: Text, model_weights: Text) -> bool:
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
    """Creates a config for LanguageModelFeaturizer.

    If CI, skips model/model_weight combinations that are too large (bert with
    LaBSE).

    Args:
        model_name: model name
        model_weights: model weights name
    """
    if skip_on_CI_with_bert(model_name, model_weights):
        pytest.skip(
            "Reason: this model is too large, loading it results in"
            "crashing of GH action workers."
        )
    config = {"model_name": model_name}
    if model_weights:
        config["model_weights"] = model_weights
    return config


def process_training_text(
    texts: List[Text],
    model_name: Text,
    model_weights: Text,
    create_language_model_featurizer: Callable[
        [Dict[Text, Any]], LanguageModelFeaturizer
    ],
    whitespace_tokenizer: WhitespaceTokenizer,
) -> List[Message]:
    """Creates a featurizer and process training data"""
    config = create_pretrained_transformers_config(model_name, model_weights)
    lm_featurizer = create_language_model_featurizer(config)

    messages = [Message.build(text=text) for text in texts]
    td = TrainingData(messages)

    whitespace_tokenizer.process_training_data(td)
    lm_featurizer.process_training_data(td)
    return messages


def process_messages(
    texts: List[Text],
    model_name: Text,
    model_weights: Text,
    create_language_model_featurizer: Callable[
        [Dict[Text, Any]], LanguageModelFeaturizer
    ],
    whitespace_tokenizer: WhitespaceTokenizer,
) -> List[Message]:
    """Creates a featurizer and processes messages"""
    config = create_pretrained_transformers_config(model_name, model_weights)
    lm_featurizer = create_language_model_featurizer(config)

    messages = []
    for text in texts:
        message = Message.build(text=text)
        whitespace_tokenizer.process([message])
        messages.append(message)
    lm_featurizer.process(messages)
    return messages


@pytest.mark.parametrize(
    "model_name, model_weights, texts, expected_shape, "
    "expected_sequence_vec, expected_cls_vec",
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
                [-0.06324312090873718, 0.4072571396827698],
                [
                    0.8041259050369263,
                    -0.08877559006214142,
                    0.9976294636726379,
                    -0.38815218210220337,
                    0.08530596643686295,
                ],
            ],
            [
                [
                    0.1720070093870163,
                    0.1511477530002594,
                    0.39497435092926025,
                    -0.5745484828948975,
                    0.05334469676017761,
                ],
                [
                    0.4095669686794281,
                    -0.11725597828626633,
                    -0.30236583948135376,
                    -0.4023253917694092,
                    0.6285617351531982,
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
                [1.7588920593261719, 2.578641176223755],
                [
                    0.7821242213249207,
                    0.6983698606491089,
                    1.5819640159606934,
                    1.891527533531189,
                    2.511735200881958,
                ],
            ],
            [
                [
                    2.168766498565674,
                    -1.5277889966964722,
                    -3.2499680519104004,
                    0.23829853534698486,
                    -1.603652000427246,
                ],
                [
                    1.643880844116211,
                    0.023089325055480003,
                    -2.497927665710449,
                    1.4621683359146118,
                    -2.5919559001922607,
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
        (
            "camembert",
            None,
            ["J'aime le camembert !"],
            [(5, 768)],
            [[0.07532623, 0.01274978, -0.08567604, 0.00386575]],
            [[0.00233287, -0.08452773, 0.0410389, 0.03026095, -0.06296296]],
        ),
    ],
)
class TestShapeValuesTrainAndProcess:
    @staticmethod
    def evaluate_message_shapes(
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
                computed_sequence_vec[: len(expected_sequence_vec[index]), 0].astype(
                    float
                ),
                expected_sequence_vec[index],
                atol=1e-4,
            )

            # Look at the first value of first five dimensions
            assert np.allclose(
                computed_sentence_vec[0][:5].astype(float),
                expected_cls_vec[index],
                atol=1e-4,
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

    @pytest.mark.timeout(120, func_only=True)
    def test_lm_featurizer_shapes_in_process_training_data(
        self,
        model_name: Text,
        model_weights: Text,
        texts: List[Text],
        expected_shape: List[Tuple[int, int]],
        expected_sequence_vec: List[List[float]],
        expected_cls_vec: List[List[float]],
        create_language_model_featurizer: Callable[
            [Dict[Text, Any]], LanguageModelFeaturizer
        ],
        whitespace_tokenizer: WhitespaceTokenizer,
    ):
        messages = process_training_text(
            texts,
            model_name,
            model_weights,
            create_language_model_featurizer,
            whitespace_tokenizer,
        )
        self.evaluate_message_shapes(
            messages, expected_shape, expected_sequence_vec, expected_cls_vec
        )

    @pytest.mark.timeout(120, func_only=True)
    def test_lm_featurizer_shapes_in_process_messages(
        self,
        model_name: Text,
        model_weights: Text,
        texts: List[Text],
        expected_shape: List[Tuple[int, int]],
        expected_sequence_vec: List[List[float]],
        expected_cls_vec: List[List[float]],
        create_language_model_featurizer: Callable[
            [Dict[Text, Any]], LanguageModelFeaturizer
        ],
        whitespace_tokenizer: WhitespaceTokenizer,
    ):
        messages = process_messages(
            texts,
            model_name,
            model_weights,
            create_language_model_featurizer,
            whitespace_tokenizer,
        )
        self.evaluate_message_shapes(
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
        (
            "camembert",
            "camembert-base",
            [
                "J'aime le camembert !",
            ],
            [[1, 1, 1, 3]],
        ),
    ],
)
class TestSubTokensTrainAndProcess:
    @staticmethod
    def check_subtokens(
        texts: List[Text],
        messages: List[Message],
        expected_number_of_sub_tokens: List[List[float]],
        whitespace_tokenizer: WhitespaceTokenizer,
    ):
        """Checks that we get the correct number of sub tokens"""
        for index, message in enumerate(messages):
            assert [
                t.get(NUMBER_OF_SUB_TOKENS) for t in message.get(TOKENS_NAMES[TEXT])
            ] == expected_number_of_sub_tokens[index]
            assert len(message.get(TOKENS_NAMES[TEXT])) == len(
                whitespace_tokenizer.tokenize(Message.build(text=texts[index]), TEXT)
            )

    @pytest.mark.timeout(120, func_only=True)
    def test_lm_featurizer_num_sub_tokens_process_training_data(
        self,
        model_name: Text,
        model_weights: Text,
        texts: List[Text],
        expected_number_of_sub_tokens: List[List[float]],
        create_language_model_featurizer: Callable[
            [Dict[Text, Any]], LanguageModelFeaturizer
        ],
        whitespace_tokenizer: WhitespaceTokenizer,
    ):
        """Tests the number of sub tokens when calling the function
        process training data"""
        messages = process_training_text(
            texts,
            model_name,
            model_weights,
            create_language_model_featurizer,
            whitespace_tokenizer,
        )
        self.check_subtokens(
            texts, messages, expected_number_of_sub_tokens, whitespace_tokenizer
        )

    @pytest.mark.timeout(120, func_only=True)
    def test_lm_featurizer_num_sub_tokens_process_messages(
        self,
        model_name: Text,
        model_weights: Text,
        texts: List[Text],
        expected_number_of_sub_tokens: List[List[float]],
        create_language_model_featurizer: Callable[
            [Dict[Text, Any]], LanguageModelFeaturizer
        ],
        whitespace_tokenizer: WhitespaceTokenizer,
    ):
        """Tests the number of sub tokens when calling the function
        process (messages)"""
        messages = process_messages(
            texts,
            model_name,
            model_weights,
            create_language_model_featurizer,
            whitespace_tokenizer,
        )
        self.check_subtokens(
            texts, messages, expected_number_of_sub_tokens, whitespace_tokenizer
        )


@pytest.mark.parametrize(
    "input_sequence_length, model_name, should_overflow",
    [(20, "bert", False), (1000, "bert", True), (1000, "xlnet", False)],
)
def test_sequence_length_overflow_train(
    input_sequence_length: int,
    model_name: Text,
    should_overflow: bool,
    create_language_model_featurizer: Callable[
        [Dict[Text, Any]], LanguageModelFeaturizer
    ],
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setattr(LanguageModelFeaturizer, "_load_model_instance", lambda _: None)
    component = create_language_model_featurizer({"model_name": model_name})
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
    create_language_model_featurizer: Callable[
        [Dict[Text, Any]], LanguageModelFeaturizer
    ],
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setattr(LanguageModelFeaturizer, "_load_model_instance", lambda _: None)
    component = create_language_model_featurizer({"model_name": model_name})
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
    create_language_model_featurizer: Callable[
        [Dict[Text, Any]], LanguageModelFeaturizer
    ],
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setattr(LanguageModelFeaturizer, "_load_model_instance", lambda _: None)
    component = create_language_model_featurizer({"model_name": "bert"})
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
    create_language_model_featurizer: Callable[
        [Dict[Text, Any]], LanguageModelFeaturizer
    ],
    whitespace_tokenizer: WhitespaceTokenizer,
):
    config = {"model_name": model_name, "model_weights": model_weights}

    featurizer = create_language_model_featurizer(config)

    text = " ".join(["hi"] * sequence_length)
    message = Message.build(text=text)
    td = TrainingData([message])
    whitespace_tokenizer.process_training_data(td)
    caplog.set_level(logging.DEBUG)
    featurizer.process([message])
    if should_overflow:
        assert "hi hi hi" in caplog.text
    assert len(message.features) >= 2


@pytest.mark.parametrize(
    "actual_sequence_length, max_input_sequence_length, zero_start_index",
    [(256, 512, 256), (700, 700, 700), (700, 512, 512)],
)
def test_attention_mask(
    actual_sequence_length: int,
    max_input_sequence_length: int,
    zero_start_index: int,
    create_language_model_featurizer: Callable[
        [Dict[Text, Any]], LanguageModelFeaturizer
    ],
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setattr(LanguageModelFeaturizer, "_load_model_instance", lambda _: None)
    component = create_language_model_featurizer({"model_name": "bert"})

    attention_mask = component._compute_attention_mask(
        [actual_sequence_length], max_input_sequence_length
    )
    mask_ones = attention_mask[0][:zero_start_index]
    mask_zeros = attention_mask[0][zero_start_index:]

    assert np.all(mask_ones == 1)
    assert np.all(mask_zeros == 0)


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
    create_language_model_featurizer: Callable[
        [Dict[Text, Any]], LanguageModelFeaturizer
    ],
):

    config = {"model_name": "bert", "model_weights": "bert-base-chinese"}

    lm_featurizer = create_language_model_featurizer(config)

    message = Message.build(text=text)
    message.set(TOKENS_NAMES[TEXT], [Token(word, start) for (word, start) in tokens])

    result, _ = lm_featurizer._tokenize_example(message, TEXT)

    assert [(token.text, token.start) for token in result] == expected_feature_tokens
