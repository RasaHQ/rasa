import copy

import pytest
import numpy as np
from typing import List, Dict, Text, Any, Optional, Tuple, Union, Callable

import rasa.model
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.training_data import util
import rasa.shared.nlu.training_data.loading
from rasa.utils.tensorflow.constants import (
    EPOCHS,
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    RENORMALIZE_CONFIDENCES,
    TRANSFORMER_SIZE,
    CONSTRAIN_SIMILARITIES,
    CHECKPOINT_MODEL,
    MODEL_CONFIDENCE,
    RANDOM_SEED,
    RANKING_LENGTH,
    LOSS_TYPE,
    HIDDEN_LAYERS_SIZES,
    LABEL,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    RUN_EAGERLY,
)
from rasa.shared.nlu.constants import (
    TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    INTENT_RESPONSE_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
from rasa.utils.tensorflow.model_data_utils import FeatureArray
from rasa.shared.nlu.training_data.loading import load_data
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.constants import (
    DEFAULT_TRANSFORMER_SIZE,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    RESPONSE_SELECTOR_PREDICTION_KEY,
    RESPONSE_SELECTOR_RESPONSES_KEY,
)


@pytest.fixture()
def response_selector_training_data() -> TrainingData:
    # use data that include some responses
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.yml"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.yml"
    )
    training_data = training_data.merge(training_data_responses)

    return training_data


@pytest.fixture()
def default_response_selector_resource() -> Resource:
    return Resource("response_selector")


@pytest.fixture
def create_response_selector(
    default_model_storage: ModelStorage,
    default_response_selector_resource: Resource,
    default_execution_context: ExecutionContext,
) -> Callable[[Dict[Text, Any]], ResponseSelector]:
    def inner(config_params: Dict[Text, Any]) -> ResponseSelector:
        return ResponseSelector.create(
            {**ResponseSelector.get_default_config(), **config_params},
            default_model_storage,
            default_response_selector_resource,
            default_execution_context,
        )

    return inner


@pytest.fixture()
def load_response_selector(
    default_model_storage: ModelStorage,
    default_response_selector_resource: Resource,
    default_execution_context: ExecutionContext,
) -> Callable[[Dict[Text, Any]], ResponseSelector]:
    def inner(config_params: Dict[Text, Any]) -> ResponseSelector:
        return ResponseSelector.load(
            {**ResponseSelector.get_default_config(), **config_params},
            default_model_storage,
            default_response_selector_resource,
            default_execution_context,
        )

    return inner


@pytest.fixture()
def train_persist_load_with_different_settings(
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    load_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    default_execution_context: ExecutionContext,
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
):
    def inner(
        pipeline: List[Dict[Text, Any]],
        config_params: Dict[Text, Any],
        should_finetune: bool,
    ):

        training_data, loaded_pipeline = train_and_preprocess(
            pipeline, "data/examples/rasa/demo-rasa.yml"
        )

        response_selector = create_response_selector(config_params)
        response_selector.train(training_data=training_data)

        if should_finetune:
            default_execution_context.is_finetuning = True

        message = Message(data={TEXT: "hello"})
        message = process_message(loaded_pipeline, message)

        message2 = copy.deepcopy(message)

        classified_message = response_selector.process([message])[0]

        loaded_selector = load_response_selector(config_params)

        classified_message2 = loaded_selector.process([message2])[0]

        assert classified_message2.fingerprint() == classified_message.fingerprint()

        return loaded_selector

    return inner


@pytest.mark.parametrize(
    "config_params",
    [
        {EPOCHS: 1, RUN_EAGERLY: True},
        {
            EPOCHS: 1,
            MASKED_LM: True,
            TRANSFORMER_SIZE: 256,
            NUM_TRANSFORMER_LAYERS: 1,
            RUN_EAGERLY: True,
        },
    ],
)
def test_train_selector(
    response_selector_training_data: TrainingData,
    config_params: Dict[Text, Any],
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    default_model_storage: ModelStorage,
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
):
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    response_selector_training_data, loaded_pipeline = train_and_preprocess(
        pipeline, response_selector_training_data
    )

    response_selector = create_response_selector(config_params)
    response_selector.train(training_data=response_selector_training_data)

    message = Message(data={TEXT: "hello"})
    message = process_message(loaded_pipeline, message)

    classified_message = response_selector.process([message])[0]

    assert classified_message is not None
    assert (
        classified_message.get("response_selector").get("all_retrieval_intents")
    ) == ["chitchat"]
    assert (
        classified_message.get("response_selector")
        .get("default")
        .get("response")
        .get("intent_response_key")
    ) is not None
    assert (
        classified_message.get("response_selector")
        .get("default")
        .get("response")
        .get("utter_action")
    ) is not None
    assert (
        classified_message.get("response_selector")
        .get("default")
        .get("response")
        .get("responses")
    ) is not None

    ranking = classified_message.get("response_selector").get("default").get("ranking")
    assert ranking is not None

    for rank in ranking:
        assert rank.get("confidence") is not None
        assert rank.get("intent_response_key") is not None


def test_preprocess_selector_multiple_retrieval_intents(
    response_selector_training_data: TrainingData,
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
):

    training_data_extra_intent = TrainingData(
        [
            Message.build(
                text="Is it possible to detect the version?", intent="faq/q1"
            ),
            Message.build(text="How can I get a new virtual env", intent="faq/q2"),
        ]
    )
    training_data = response_selector_training_data.merge(training_data_extra_intent)

    response_selector = create_response_selector({})

    response_selector.preprocess_train_data(training_data)

    assert sorted(response_selector.all_retrieval_intents) == ["chitchat", "faq"]


@pytest.mark.parametrize(
    "use_text_as_label, label_values",
    [
        [False, ["chitchat/ask_name", "chitchat/ask_weather"]],
        [True, ["I am Mr. Bot", "It's sunny where I live"]],
    ],
)
def test_ground_truth_for_training(
    use_text_as_label,
    label_values,
    response_selector_training_data: TrainingData,
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
):
    response_selector = create_response_selector(
        {"use_text_as_label": use_text_as_label}
    )
    response_selector.preprocess_train_data(response_selector_training_data)

    assert response_selector.responses == response_selector_training_data.responses
    assert (
        sorted(list(response_selector.index_label_id_mapping.values())) == label_values
    )


@pytest.mark.parametrize(
    "predicted_label, train_on_text, resolved_intent_response_key",
    [
        ["chitchat/ask_name", False, "chitchat/ask_name"],
        ["It's sunny where I live", True, "chitchat/ask_weather"],
    ],
)
def test_resolve_intent_response_key_from_label(
    predicted_label: Text,
    train_on_text: bool,
    resolved_intent_response_key: Text,
    response_selector_training_data: TrainingData,
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
):

    response_selector = create_response_selector({"use_text_as_label": train_on_text})
    response_selector.preprocess_train_data(response_selector_training_data)

    label_intent_response_key = response_selector._resolve_intent_response_key(
        {"id": hash(predicted_label), "name": predicted_label}
    )
    assert resolved_intent_response_key == label_intent_response_key
    assert (
        response_selector.responses[
            util.intent_response_key_to_template_key(label_intent_response_key)
        ]
        == response_selector_training_data.responses[
            util.intent_response_key_to_template_key(resolved_intent_response_key)
        ]
    )


def test_train_model_checkpointing(
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    default_model_storage: ModelStorage,
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
):
    pipeline = [
        {"component": WhitespaceTokenizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 3,
            "max_ngram": 17,
            "max_features": 10,
            "min_df": 5,
        },
    ]

    training_data, loaded_pipeline = train_and_preprocess(
        pipeline, "data/test_selectors"
    )

    config_params = {
        EPOCHS: 2,
        MODEL_CONFIDENCE: "softmax",
        CONSTRAIN_SIMILARITIES: True,
        CHECKPOINT_MODEL: True,
        EVAL_NUM_EPOCHS: 1,
        EVAL_NUM_EXAMPLES: 10,
        RUN_EAGERLY: True,
    }

    response_selector = create_response_selector(config_params)
    assert response_selector.component_config[CHECKPOINT_MODEL]

    resource = response_selector.train(training_data=training_data)

    with default_model_storage.read_from(resource) as model_dir:
        all_files = list(model_dir.rglob("*.*"))
        assert any(["from_checkpoint" in str(filename) for filename in all_files])


@pytest.mark.skip_on_windows
def test_train_persist_load(
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    load_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    default_execution_context: ExecutionContext,
    train_persist_load_with_different_settings,
):

    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    config_params = {EPOCHS: 1, RUN_EAGERLY: True}

    train_persist_load_with_different_settings(pipeline, config_params, False)

    train_persist_load_with_different_settings(pipeline, config_params, True)


async def test_process_gives_diagnostic_data(
    default_execution_context: ExecutionContext,
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
):
    """Tests if processing a message returns attention weights as numpy array."""
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    config_params = {EPOCHS: 1, RUN_EAGERLY: True}

    importer = RasaFileImporter(
        config_file="data/test_response_selector_bot/config.yml",
        domain_path="data/test_response_selector_bot/domain.yml",
        training_data_paths=[
            "data/test_response_selector_bot/data/rules.yml",
            "data/test_response_selector_bot/data/stories.yml",
            "data/test_response_selector_bot/data/nlu.yml",
        ],
    )
    training_data = importer.get_nlu_data()

    training_data, loaded_pipeline = train_and_preprocess(pipeline, training_data)

    default_execution_context.should_add_diagnostic_data = True

    response_selector = create_response_selector(config_params)
    response_selector.train(training_data=training_data)

    message = Message(data={TEXT: "hello"})
    message = process_message(loaded_pipeline, message)

    classified_message = response_selector.process([message])[0]
    diagnostic_data = classified_message.get(DIAGNOSTIC_DATA)

    assert isinstance(diagnostic_data, dict)
    for _, values in diagnostic_data.items():
        assert "text_transformed" in values
        assert isinstance(values.get("text_transformed"), np.ndarray)
        # The `attention_weights` key should exist, regardless of there
        # being a transformer
        assert "attention_weights" in values
        # By default, ResponseSelector has `number_of_transformer_layers = 0`
        # in which case the attention weights should be None.
        assert values.get("attention_weights") is None


@pytest.mark.parametrize(
    "classifier_params",
    [({LOSS_TYPE: "margin", RANDOM_SEED: 42, EPOCHS: 1, RUN_EAGERLY: True})],
)
async def test_margin_loss_is_not_normalized(
    classifier_params: Dict[Text, int],
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
):
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    training_data, loaded_pipeline = train_and_preprocess(
        pipeline, "data/test_selectors"
    )

    response_selector = create_response_selector(classifier_params)
    response_selector.train(training_data=training_data)

    message = Message(data={TEXT: "hello"})
    message = process_message(loaded_pipeline, message)

    classified_message = response_selector.process([message])[0]

    response_ranking = (
        classified_message.get("response_selector").get("default").get("ranking")
    )

    # check that output was not normalized
    assert [item["confidence"] for item in response_ranking] != pytest.approx(1)

    # check that the output was correctly truncated
    assert len(response_ranking) == 9


@pytest.mark.parametrize(
    "classifier_params, output_length, sums_up_to_1",
    [
        ({}, 9, True),
        ({EPOCHS: 1, RUN_EAGERLY: True}, 9, True),
        ({RANKING_LENGTH: 2}, 2, False),
        ({RANKING_LENGTH: 2, RENORMALIZE_CONFIDENCES: True}, 2, True),
    ],
)
async def test_softmax_ranking(
    classifier_params: Dict[Text, int],
    output_length: int,
    sums_up_to_1: bool,
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
):
    classifier_params[RANDOM_SEED] = 42
    classifier_params[EPOCHS] = 1
    classifier_params[RUN_EAGERLY] = True

    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    training_data, loaded_pipeline = train_and_preprocess(
        pipeline, "data/test_selectors"
    )

    response_selector = create_response_selector(classifier_params)
    response_selector.train(training_data=training_data)

    message = Message(data={TEXT: "hello"})
    message = process_message(loaded_pipeline, message)

    classified_message = response_selector.process([message])[0]

    response_ranking = (
        classified_message.get("response_selector").get("default").get("ranking")
    )
    # check that the output was correctly truncated after normalization
    assert len(response_ranking) == output_length
    output_sums_to_1 = sum(
        [intent.get("confidence") for intent in response_ranking]
    ) == pytest.approx(1)
    assert output_sums_to_1 == sums_up_to_1


@pytest.mark.parametrize(
    "config, should_raise_warning",
    [
        # hidden layers left at defaults
        ({}, False),
        ({NUM_TRANSFORMER_LAYERS: 5}, True),
        ({NUM_TRANSFORMER_LAYERS: 0}, False),
        ({NUM_TRANSFORMER_LAYERS: -1}, False),
        # hidden layers explicitly enabled
        ({HIDDEN_LAYERS_SIZES: {TEXT: [10], LABEL: [11]}}, False),
        (
            {NUM_TRANSFORMER_LAYERS: 5, HIDDEN_LAYERS_SIZES: {TEXT: [10], LABEL: [11]}},
            True,
        ),
        (
            {NUM_TRANSFORMER_LAYERS: 0, HIDDEN_LAYERS_SIZES: {TEXT: [10], LABEL: [11]}},
            False,
        ),
        (
            {
                NUM_TRANSFORMER_LAYERS: -1,
                HIDDEN_LAYERS_SIZES: {TEXT: [10], LABEL: [11]},
            },
            False,
        ),
        # hidden layers explicitly disabled
        ({HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []}}, False),
        (
            {NUM_TRANSFORMER_LAYERS: 5, HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []}},
            False,
        ),
        (
            {NUM_TRANSFORMER_LAYERS: 0, HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []}},
            False,
        ),
        (
            {NUM_TRANSFORMER_LAYERS: -1, HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []}},
            False,
        ),
    ],
)
def test_warning_when_transformer_and_hidden_layers_enabled(
    config: Dict[Text, Union[int, Dict[Text, List[int]]]],
    should_raise_warning: bool,
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
):
    """ResponseSelector recommends disabling hidden layers if transformer is enabled."""
    with pytest.warns(UserWarning) as records:
        _ = create_response_selector(config)
    warning_str = "We recommend to disable the hidden layers when using a transformer"

    if should_raise_warning:
        assert len(records) > 0
        # Check all warnings since there may be multiple other warnings we don't care
        # about in this test case.
        assert any(warning_str in record.message.args[0] for record in records)
    else:
        # Check all warnings since there may be multiple other warnings we don't care
        # about in this test case.
        assert not any(warning_str in record.message.args[0] for record in records)


@pytest.mark.parametrize(
    "config, should_set_default_transformer_size",
    [
        # transformer enabled
        ({NUM_TRANSFORMER_LAYERS: 5}, True),
        ({TRANSFORMER_SIZE: 0, NUM_TRANSFORMER_LAYERS: 5}, True),
        ({TRANSFORMER_SIZE: -1, NUM_TRANSFORMER_LAYERS: 5}, True),
        ({TRANSFORMER_SIZE: None, NUM_TRANSFORMER_LAYERS: 5}, True),
        ({TRANSFORMER_SIZE: 10, NUM_TRANSFORMER_LAYERS: 5}, False),
        # transformer disabled (by default)
        ({}, False),
        ({TRANSFORMER_SIZE: 0}, False),
        ({TRANSFORMER_SIZE: -1}, False),
        ({TRANSFORMER_SIZE: None}, False),
        ({TRANSFORMER_SIZE: 10}, False),
        # transformer disabled explicitly
        ({NUM_TRANSFORMER_LAYERS: 0}, False),
        ({TRANSFORMER_SIZE: 0, NUM_TRANSFORMER_LAYERS: 0}, False),
        ({TRANSFORMER_SIZE: -1, NUM_TRANSFORMER_LAYERS: 0}, False),
        ({TRANSFORMER_SIZE: None, NUM_TRANSFORMER_LAYERS: 0}, False),
        ({TRANSFORMER_SIZE: 10, NUM_TRANSFORMER_LAYERS: 0}, False),
    ],
)
def test_sets_integer_transformer_size_when_needed(
    config: Dict[Text, Optional[int]],
    should_set_default_transformer_size: bool,
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
):
    """ResponseSelector ensures sensible transformer size when transformer enabled."""
    with pytest.warns(UserWarning) as records:
        selector = create_response_selector(config)

    warning_str = f"positive size is required when using `{NUM_TRANSFORMER_LAYERS} > 0`"

    if should_set_default_transformer_size:
        assert len(records) > 0
        # check that the specific warning was raised
        assert any(warning_str in record.message.args[0] for record in records)
        # check that transformer size got set to the new default
        assert selector.component_config[TRANSFORMER_SIZE] == DEFAULT_TRANSFORMER_SIZE
    else:
        # check that the specific warning was not raised
        assert not any(warning_str in record.message.args[0] for record in records)
        # check that transformer size was not changed
        assert selector.component_config[TRANSFORMER_SIZE] == config.get(
            TRANSFORMER_SIZE, None  # None is the default transformer size
        )


def test_transformer_size_gets_corrected(train_persist_load_with_different_settings):
    """Tests that the default value of `transformer_size` which is `None` is
    corrected if transformer layers are enabled in `ResponseSelector`.
    """
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    config_params = {EPOCHS: 1, NUM_TRANSFORMER_LAYERS: 1, RUN_EAGERLY: True}

    selector = train_persist_load_with_different_settings(
        pipeline, config_params, False
    )
    assert selector.component_config[TRANSFORMER_SIZE] == DEFAULT_TRANSFORMER_SIZE


async def test_process_unfeaturized_input(
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
):
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    training_data, loaded_pipeline = train_and_preprocess(
        pipeline, "data/test_selectors"
    )
    response_selector = create_response_selector({EPOCHS: 1, RUN_EAGERLY: True})
    response_selector.train(training_data=training_data)

    message_text = "message text"
    unfeaturized_message = Message(data={TEXT: message_text})
    classified_message = response_selector.process([unfeaturized_message])[0]
    output = (
        classified_message.get(RESPONSE_SELECTOR_PROPERTY_NAME)
        .get(RESPONSE_SELECTOR_DEFAULT_INTENT)
        .get(RESPONSE_SELECTOR_PREDICTION_KEY)
    )

    assert classified_message.get(TEXT) == message_text
    assert not output.get(RESPONSE_SELECTOR_RESPONSES_KEY)
    assert output.get(PREDICTED_CONFIDENCE_KEY) == 0.0
    assert not output.get(INTENT_RESPONSE_KEY)


@pytest.mark.timeout(120)
async def test_adjusting_layers_incremental_training(
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    load_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
):
    """Tests adjusting sparse layers of `ResponseSelector` to increased sparse
    feature sizes during incremental training.

    Testing is done by checking the layer sizes.
    Checking if they were replaced correctly is also important
    and is done in `test_replace_dense_for_sparse_layers`
    in `test_rasa_layers.py`.
    """
    iter1_data_path = "data/test_incremental_training/iter1/"
    iter2_data_path = "data/test_incremental_training/"
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": LexicalSyntacticFeaturizer},
        {"component": RegexFeaturizer},
        {"component": CountVectorsFeaturizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
    ]
    training_data, loaded_pipeline = train_and_preprocess(pipeline, iter1_data_path)
    response_selector = create_response_selector({EPOCHS: 1, RUN_EAGERLY: True})
    response_selector.train(training_data=training_data)

    old_data_signature = response_selector.model.data_signature
    old_predict_data_signature = response_selector.model.predict_data_signature

    message = Message(data={TEXT: "Rasa is great!"})
    message = process_message(loaded_pipeline, message)

    message2 = copy.deepcopy(message)

    classified_message = response_selector.process([message])[0]

    old_sparse_feature_sizes = classified_message.get_sparse_feature_sizes(
        attribute=TEXT
    )

    initial_rs_layers = response_selector.model._tf_layers[
        "sequence_layer.text"
    ]._tf_layers["feature_combining"]
    initial_rs_sequence_layer = initial_rs_layers._tf_layers[
        "sparse_dense.sequence"
    ]._tf_layers["sparse_to_dense"]
    initial_rs_sentence_layer = initial_rs_layers._tf_layers[
        "sparse_dense.sentence"
    ]._tf_layers["sparse_to_dense"]

    initial_rs_sequence_size = initial_rs_sequence_layer.get_kernel().shape[0]
    initial_rs_sentence_size = initial_rs_sentence_layer.get_kernel().shape[0]
    assert initial_rs_sequence_size == sum(
        old_sparse_feature_sizes[FEATURE_TYPE_SEQUENCE]
    )
    assert initial_rs_sentence_size == sum(
        old_sparse_feature_sizes[FEATURE_TYPE_SENTENCE]
    )

    loaded_selector = load_response_selector({EPOCHS: 1, RUN_EAGERLY: True})

    classified_message2 = loaded_selector.process([message2])[0]

    assert classified_message2.fingerprint() == classified_message.fingerprint()

    training_data2, loaded_pipeline2 = train_and_preprocess(pipeline, iter2_data_path)

    response_selector.train(training_data=training_data2)

    new_message = Message.build(text="Rasa is great!")
    new_message = process_message(loaded_pipeline2, new_message)

    classified_new_message = response_selector.process([new_message])[0]
    new_sparse_feature_sizes = classified_new_message.get_sparse_feature_sizes(
        attribute=TEXT
    )

    final_rs_layers = response_selector.model._tf_layers[
        "sequence_layer.text"
    ]._tf_layers["feature_combining"]
    final_rs_sequence_layer = final_rs_layers._tf_layers[
        "sparse_dense.sequence"
    ]._tf_layers["sparse_to_dense"]
    final_rs_sentence_layer = final_rs_layers._tf_layers[
        "sparse_dense.sentence"
    ]._tf_layers["sparse_to_dense"]

    final_rs_sequence_size = final_rs_sequence_layer.get_kernel().shape[0]
    final_rs_sentence_size = final_rs_sentence_layer.get_kernel().shape[0]
    assert final_rs_sequence_size == sum(
        new_sparse_feature_sizes[FEATURE_TYPE_SEQUENCE]
    )
    assert final_rs_sentence_size == sum(
        new_sparse_feature_sizes[FEATURE_TYPE_SENTENCE]
    )
    # check if the data signatures were correctly updated
    new_data_signature = response_selector.model.data_signature
    new_predict_data_signature = response_selector.model.predict_data_signature
    iter2_data = load_data(iter2_data_path)
    expected_sequence_lengths = len(
        [
            message
            for message in iter2_data.training_examples
            if message.get(INTENT_RESPONSE_KEY)
        ]
    )

    def test_data_signatures(
        new_signature: Dict[Text, Dict[Text, List[FeatureArray]]],
        old_signature: Dict[Text, Dict[Text, List[FeatureArray]]],
    ):
        # Wherever attribute / feature_type signature is not
        # expected to change, directly compare it to old data signature.
        # Else compute its expected signature and compare
        attributes_expected_to_change = [TEXT]
        feature_types_expected_to_change = [
            FEATURE_TYPE_SEQUENCE,
            FEATURE_TYPE_SENTENCE,
        ]

        for attribute, signatures in new_signature.items():

            for feature_type, feature_signatures in signatures.items():

                if feature_type == "sequence_lengths":
                    assert feature_signatures[0].units == expected_sequence_lengths

                elif feature_type not in feature_types_expected_to_change:
                    assert feature_signatures == old_signature.get(attribute).get(
                        feature_type
                    )
                else:
                    for index, feature_signature in enumerate(feature_signatures):
                        if (
                            feature_signature.is_sparse
                            and attribute in attributes_expected_to_change
                        ):
                            assert feature_signature.units == sum(
                                new_sparse_feature_sizes.get(feature_type)
                            )
                        else:
                            # dense signature or attributes that are not
                            # expected to change can be compared directly
                            assert (
                                feature_signature.units
                                == old_signature.get(attribute)
                                .get(feature_type)[index]
                                .units
                            )

    test_data_signatures(new_data_signature, old_data_signature)
    test_data_signatures(new_predict_data_signature, old_predict_data_signature)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "iter1_path, iter2_path, should_raise_exception",
    [
        (
            "data/test_incremental_training/",
            "data/test_incremental_training/iter1",
            True,
        ),
        (
            "data/test_incremental_training/iter1",
            "data/test_incremental_training/",
            False,
        ),
    ],
)
async def test_sparse_feature_sizes_decreased_incremental_training(
    iter1_path: Text,
    iter2_path: Text,
    should_raise_exception: bool,
    create_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    load_response_selector: Callable[[Dict[Text, Any]], ResponseSelector],
    default_execution_context: ExecutionContext,
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
):
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": LexicalSyntacticFeaturizer},
        {"component": RegexFeaturizer},
        {"component": CountVectorsFeaturizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
    ]
    training_data, loaded_pipeline = train_and_preprocess(pipeline, iter1_path)

    response_selector = create_response_selector({EPOCHS: 1, RUN_EAGERLY: True})
    response_selector.train(training_data=training_data)

    message = Message(data={TEXT: "Rasa is great!"})
    message = process_message(loaded_pipeline, message)

    message2 = copy.deepcopy(message)

    classified_message = response_selector.process([message])[0]

    default_execution_context.is_finetuning = True

    loaded_selector = load_response_selector({EPOCHS: 1, RUN_EAGERLY: True})

    classified_message2 = loaded_selector.process([message2])[0]

    assert classified_message2.fingerprint() == classified_message.fingerprint()

    if should_raise_exception:
        with pytest.raises(Exception) as exec_info:
            training_data2, loaded_pipeline2 = train_and_preprocess(
                pipeline, iter2_path
            )
            loaded_selector.train(training_data=training_data2)
        assert "Sparse feature sizes have decreased" in str(exec_info.value)
    else:
        training_data2, loaded_pipeline2 = train_and_preprocess(pipeline, iter2_path)
        loaded_selector.train(training_data=training_data2)
        assert loaded_selector.model
