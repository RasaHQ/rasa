import copy
from pathlib import Path

import pytest
import numpy as np
from typing import List, Dict, Text, Any, Optional, Union
from unittest.mock import Mock
from _pytest.monkeypatch import MonkeyPatch

import rasa.model
import rasa.nlu.train
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu import registry
from rasa.nlu.components import ComponentBuilder
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.training_data import util
from rasa.nlu.config import RasaNLUModelConfig
import rasa.shared.nlu.training_data.loading
from rasa.nlu.train import Interpreter
from rasa.utils.tensorflow.constants import (
    EPOCHS,
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    CONSTRAIN_SIMILARITIES,
    CHECKPOINT_MODEL,
    MODEL_CONFIDENCE,
    RANDOM_SEED,
    RANKING_LENGTH,
    LOSS_TYPE,
    HIDDEN_LAYERS_SIZES,
    LABEL,
)
from rasa.utils import train_utils
from rasa.shared.nlu.constants import (
    TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    INTENT_RESPONSE_KEY,
)
from rasa.utils.tensorflow.model_data_utils import FeatureArray
from rasa.shared.nlu.training_data.loading import load_data
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.nlu.selectors.response_selector import ResponseSelectorGraphComponent
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


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


def create_response_selector(
    config_params: Dict[Text, Any],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> ResponseSelectorGraphComponent:
    return ResponseSelectorGraphComponent.create(
        {**ResponseSelectorGraphComponent.get_default_config(), **config_params},
        default_model_storage,
        Resource("response_selector"),
        default_execution_context,
    )


@pytest.mark.parametrize(
    "pipeline, config_params",
    [
        (
            [{"name": "WhitespaceTokenizer"}, {"name": "CountVectorsFeaturizer"},],
            {EPOCHS: 1},
        ),
        (
            [{"name": "WhitespaceTokenizer"}, {"name": "CountVectorsFeaturizer"},],
            {
                EPOCHS: 1,
                MASKED_LM: True,
                TRANSFORMER_SIZE: 256,
                NUM_TRANSFORMER_LAYERS: 1,
            },
        ),
    ],
)
def test_train_selector(
    response_selector_training_data: TrainingData,
    pipeline: List[Dict[Text, Text]],
    config_params: Dict[Text, Any],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    loaded_pipeline = [
        registry.get_component_class(component.pop("name"))(component)
        for component in copy.deepcopy(pipeline)
    ]

    for component in loaded_pipeline:
        component.train(response_selector_training_data)

    response_selector = create_response_selector(
        config_params, default_model_storage, default_execution_context,
    )

    response_selector.train(training_data=response_selector_training_data)

    message = Message(data={TEXT: "hello"})
    for component in loaded_pipeline:
        component.process(message)

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
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
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

    response_selector = create_response_selector(
        {}, default_model_storage, default_execution_context
    )

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
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    response_selector = create_response_selector(
        {"use_text_as_label": use_text_as_label},
        default_model_storage,
        default_execution_context,
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
    predicted_label,
    train_on_text,
    resolved_intent_response_key,
    response_selector_training_data,
    default_model_storage,
    default_execution_context,
):

    response_selector = create_response_selector(
        {"use_text_as_label": train_on_text},
        default_model_storage,
        default_execution_context,
    )
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


async def test_train_model_checkpointing(
    component_builder: ComponentBuilder, tmpdir: Path
):
    from pathlib import Path

    model_name = "rs-checkpointed-model"
    best_model_file = Path(str(tmpdir), model_name)
    assert not best_model_file.exists()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {
                    "name": "CountVectorsFeaturizer",
                    "analyzer": "char_wb",
                    "min_ngram": 3,
                    "max_ngram": 17,
                    "max_features": 10,
                    "min_df": 5,
                },
                {
                    "name": "ResponseSelector",
                    EPOCHS: 5,
                    MODEL_CONFIDENCE: "linear_norm",
                    CONSTRAIN_SIMILARITIES: True,
                    CHECKPOINT_MODEL: True,
                },
            ],
            "language": "en",
        }
    )

    await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data="data/test_selectors",
        component_builder=component_builder,
        fixed_model_name=model_name,
    )

    assert best_model_file.exists()

    """
    Tricky to validate the *exact* number of files that should be there, however there
    must be at least the following:
        - metadata.json
        - checkpoint
        - component_1_CountVectorsFeaturizer (as per the pipeline above)
        - component_2_ResponseSelector files (more than 1 file)
    """
    all_files = list(best_model_file.rglob("*.*"))
    assert len(all_files) > 4


def _train_persist_load_with_different_settings(
    pipeline: List[Dict[Text, Any]],
    config_params: Dict[Text, Any],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    should_finetune: bool,
):
    loaded_pipeline = [
        registry.get_component_class(component.pop("name"))(component)
        for component in copy.deepcopy(pipeline)
    ]

    importer = RasaFileImporter(
        training_data_paths=["data/examples/rasa/demo-rasa.yml"]
    )
    training_data = importer.get_nlu_data()

    for component in loaded_pipeline:
        component.train(training_data)

    if should_finetune:
        default_execution_context.is_finetuning = True

    response_selector = create_response_selector(
        config_params, default_model_storage, default_execution_context
    )
    response_selector.train(training_data=training_data)

    message = Message(data={TEXT: "Rasa is great!"})

    for component in loaded_pipeline:
        component.process(message)

    message2 = copy.deepcopy(message)

    classified_message = response_selector.process([message])[0]

    loaded_selector = ResponseSelectorGraphComponent.load(
        {**ResponseSelectorGraphComponent.get_default_config(), **config_params},
        default_model_storage,
        Resource("response_selector"),
        default_execution_context,
    )

    classified_message2 = loaded_selector.process([message2])[0]

    assert classified_message2.fingerprint() == classified_message.fingerprint()


@pytest.mark.skip_on_windows
def test_train_persist_load(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext,
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
    ]
    config_params = {EPOCHS: 1}

    _train_persist_load_with_different_settings(
        pipeline, config_params, default_model_storage, default_execution_context, False
    )

    # _train_persist_load_with_different_settings(
    #     pipeline,
    #     config_params,
    #     default_model_storage,
    #     default_execution_context,
    #     True
    # )


async def test_process_gives_diagnostic_data(
    response_selector_interpreter: Interpreter,
):
    """Tests if processing a message returns attention weights as numpy array."""
    interpreter = response_selector_interpreter

    message = Message(data={TEXT: "hello"})
    for component in interpreter.pipeline:
        component.process(message)

    diagnostic_data = message.get(DIAGNOSTIC_DATA)

    # The last component is ResponseSelector, which should add diagnostic data
    name = f"component_{len(interpreter.pipeline) - 1}_ResponseSelector"
    assert isinstance(diagnostic_data, dict)
    assert name in diagnostic_data
    assert "text_transformed" in diagnostic_data[name]
    assert isinstance(diagnostic_data[name].get("text_transformed"), np.ndarray)
    # The `attention_weights` key should exist, regardless of there being a transformer
    assert "attention_weights" in diagnostic_data[name]
    # By default, ResponseSelector has `number_of_transformer_layers = 0`
    # in which case the attention weights should be None.
    assert diagnostic_data[name].get("attention_weights") is None


@pytest.mark.parametrize(
    "classifier_params, output_length",
    [({RANDOM_SEED: 42, EPOCHS: 1, MODEL_CONFIDENCE: "linear_norm"}, 9)],
)
async def test_cross_entropy_with_linear_norm(
    classifier_params: Dict[Text, Any],
    output_length: int,
    monkeypatch: MonkeyPatch,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
    ]
    loaded_pipeline = [
        registry.get_component_class(component.pop("name"))(component)
        for component in copy.deepcopy(pipeline)
    ]

    importer = RasaFileImporter(training_data_paths=["data/test_selectors"])
    training_data = importer.get_nlu_data()

    for component in loaded_pipeline:
        component.train(training_data)

    response_selector = create_response_selector(
        classifier_params, default_model_storage, default_execution_context
    )
    response_selector.train(training_data=training_data)

    message = Message(data={TEXT: "hello"})

    for component in loaded_pipeline:
        component.process(message)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    classified_message = response_selector.process([message])[0]

    response_ranking = (
        classified_message.get("response_selector").get("default").get("ranking")
    )

    # check that the output was correctly truncated
    assert len(response_ranking) == output_length

    response_confidences = [response.get("confidence") for response in response_ranking]

    # check whether normalization had the expected effect
    output_sums_to_1 = sum(response_confidences) == pytest.approx(1)
    assert output_sums_to_1

    # normalize shouldn't have been called
    mock.normalize.assert_not_called()


@pytest.mark.parametrize(
    "classifier_params", [({LOSS_TYPE: "margin", RANDOM_SEED: 42, EPOCHS: 1})],
)
async def test_margin_loss_is_not_normalized(
    monkeypatch: MonkeyPatch,
    classifier_params: Dict[Text, int],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
    ]
    loaded_pipeline = [
        registry.get_component_class(component.pop("name"))(component)
        for component in copy.deepcopy(pipeline)
    ]

    importer = RasaFileImporter(training_data_paths=["data/test_selectors"])
    training_data = importer.get_nlu_data()

    for component in loaded_pipeline:
        component.train(training_data)

    response_selector = create_response_selector(
        classifier_params, default_model_storage, default_execution_context
    )
    response_selector.train(training_data=training_data)

    message = Message(data={TEXT: "hello"})

    for component in loaded_pipeline:
        component.process(message)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    classified_message = response_selector.process([message])[0]

    response_ranking = (
        classified_message.get("response_selector").get("default").get("ranking")
    )

    # check that the output was not normalized
    mock.normalize.assert_not_called()

    # check that the output was correctly truncated
    assert len(response_ranking) == 9


@pytest.mark.parametrize(
    "classifier_params, data_path, output_length",
    [
        ({RANDOM_SEED: 42, EPOCHS: 1}, "data/test_selectors", 9),
        ({RANDOM_SEED: 42, RANKING_LENGTH: 0, EPOCHS: 1}, "data/test_selectors", 9),
        ({RANDOM_SEED: 42, RANKING_LENGTH: 2, EPOCHS: 1}, "data/test_selectors", 2),
    ],
)
async def test_softmax_ranking(
    classifier_params: Dict[Text, int],
    data_path: Text,
    output_length: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
    ]
    loaded_pipeline = [
        registry.get_component_class(component.pop("name"))(component)
        for component in copy.deepcopy(pipeline)
    ]

    importer = RasaFileImporter(training_data_paths=["data/test_selectors"])
    training_data = importer.get_nlu_data()

    for component in loaded_pipeline:
        component.train(training_data)

    response_selector = create_response_selector(
        classifier_params, default_model_storage, default_execution_context
    )
    response_selector.train(training_data=training_data)

    message = Message(data={TEXT: "hello"})

    for component in loaded_pipeline:
        component.process(message)

    classified_message = response_selector.process([message])[0]

    response_ranking = (
        classified_message.get("response_selector").get("default").get("ranking")
    )
    # check that the output was correctly truncated after normalization
    assert len(response_ranking) == output_length


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
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    """ResponseSelector recommends disabling hidden layers if transformer is enabled."""
    with pytest.warns(UserWarning) as records:
        _ = create_response_selector(
            config, default_model_storage, default_execution_context
        )
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
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    """ResponseSelector ensures sensible transformer size when transformer enabled."""
    default_transformer_size = 256
    with pytest.warns(UserWarning) as records:
        selector = create_response_selector(
            config, default_model_storage, default_execution_context
        )

    warning_str = f"positive size is required when using `{NUM_TRANSFORMER_LAYERS} > 0`"

    if should_set_default_transformer_size:
        assert len(records) > 0
        # check that the specific warning was raised
        assert any(warning_str in record.message.args[0] for record in records)
        # check that transformer size got set to the new default
        assert selector.component_config[TRANSFORMER_SIZE] == default_transformer_size
    else:
        # check that the specific warning was not raised
        assert not any(warning_str in record.message.args[0] for record in records)
        # check that transformer size was not changed
        assert selector.component_config[TRANSFORMER_SIZE] == config.get(
            TRANSFORMER_SIZE, None  # None is the default transformer size
        )


@pytest.mark.timeout(120)
async def test_adjusting_layers_incremental_training(
    component_builder: ComponentBuilder, tmpdir: Path
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
        {"name": "WhitespaceTokenizer"},
        {"name": "LexicalSyntacticFeaturizer"},
        {"name": "RegexFeaturizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "CountVectorsFeaturizer",
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
        {"name": "ResponseSelector", EPOCHS: 1},
    ]
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (_, trained, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data=iter1_data_path,
        component_builder=component_builder,
    )
    assert trained.pipeline
    old_data_signature = trained.pipeline[-1].model.data_signature
    old_predict_data_signature = trained.pipeline[-1].model.predict_data_signature
    message = Message.build(text="Rasa is great!")
    trained.featurize_message(message)
    old_sparse_feature_sizes = message.get_sparse_feature_sizes(attribute=TEXT)
    initial_rs_layers = (
        trained.pipeline[-1]
        .model._tf_layers["sequence_layer.text"]
        ._tf_layers["feature_combining"]
    )
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

    loaded = Interpreter.load(persisted_path, component_builder, new_config=_config,)
    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") == trained.parse("Rasa is great!")
    (_, trained, _) = await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data=iter2_data_path,
        component_builder=component_builder,
        model_to_finetune=loaded,
    )
    assert trained.pipeline
    message = Message.build(text="Rasa is great!")
    trained.featurize_message(message)
    new_sparse_feature_sizes = message.get_sparse_feature_sizes(attribute=TEXT)

    final_rs_layers = (
        trained.pipeline[-1]
        .model._tf_layers["sequence_layer.text"]
        ._tf_layers["feature_combining"]
    )
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
    new_data_signature = trained.pipeline[-1].model.data_signature
    new_predict_data_signature = trained.pipeline[-1].model.predict_data_signature
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
    component_builder: ComponentBuilder,
    tmpdir: Path,
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "LexicalSyntacticFeaturizer"},
        {"name": "RegexFeaturizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "CountVectorsFeaturizer",
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
        {"name": "ResponseSelector", EPOCHS: 1},
    ]
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (_, trained, persisted_path) = await rasa.nlu.train.train(
        _config, path=str(tmpdir), data=iter1_path, component_builder=component_builder,
    )
    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder, new_config=_config,)
    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") == trained.parse("Rasa is great!")
    if should_raise_exception:
        with pytest.raises(Exception) as exec_info:
            (_, trained, _) = await rasa.nlu.train.train(
                _config,
                path=str(tmpdir),
                data=iter2_path,
                component_builder=component_builder,
                model_to_finetune=loaded,
            )
        assert "Sparse feature sizes have decreased" in str(exec_info.value)
    else:
        (_, trained, _) = await rasa.nlu.train.train(
            _config,
            path=str(tmpdir),
            data=iter2_path,
            component_builder=component_builder,
            model_to_finetune=loaded,
        )
        assert trained.pipeline
