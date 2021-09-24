import copy
import itertools
from pathlib import Path

import pytest
import numpy as np
from typing import List, Dict, Text, Any, Optional, Tuple, Union
from unittest.mock import Mock
from _pytest.monkeypatch import MonkeyPatch

import rasa.model
import rasa.nlu.train
from rasa.nlu.components import ComponentBuilder
from rasa.shared.nlu.training_data import util
from rasa.nlu.config import RasaNLUModelConfig
import rasa.shared.nlu.training_data.loading
from rasa.nlu.train import Trainer, Interpreter
from rasa.utils.tensorflow.constants import (
    EPOCHS,
    EVAL_NUM_EPOCHS,
    FEATURIZERS,
    IDS,
    MASK,
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    RETRIEVAL_INTENT,
    SENTENCE,
    SEQUENCE,
    SEQUENCE_LENGTH,
    TRANSFORMER_SIZE,
    CONSTRAIN_SIMILARITIES,
    CHECKPOINT_MODEL,
    MODEL_CONFIDENCE,
    RANDOM_SEED,
    RANKING_LENGTH,
    LOSS_TYPE,
    HIDDEN_LAYERS_SIZES,
    LABEL,
    USE_TEXT_AS_LABEL,
)
from rasa.utils import train_utils
from rasa.shared.nlu.constants import (
    INTENT,
    RESPONSE,
    TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    INTENT_RESPONSE_KEY,
)
from rasa.nlu.selectors.response_selector import (
    ResponseSelector,
    LABEL_SUB_KEY,
    LABEL_KEY,
)
from rasa.utils.tensorflow.model_data_utils import FeatureArray
from rasa.shared.nlu.training_data.loading import load_data
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from tests.nlu.classifiers.test_diet_classifier import (
    FeatureGenerator,
    FeaturizerDescription,
    SimpleIntentClassificationTestCaseWithEntities,
    as_pipeline,
)


@pytest.mark.parametrize(
    "pipeline",
    [
        [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {"name": "ResponseSelector", EPOCHS: 1},
        ],
        [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {
                "name": "ResponseSelector",
                EPOCHS: 1,
                MASKED_LM: True,
                TRANSFORMER_SIZE: 256,
                NUM_TRANSFORMER_LAYERS: 1,
            },
        ],
        [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {
                "name": "ResponseSelector",
                EPOCHS: 1,
                USE_TEXT_AS_LABEL: True,
                NUM_TRANSFORMER_LAYERS: 0,
            },
        ],
    ],
)
def test_train_selector(pipeline, component_builder, tmpdir):
    # use data that include some responses
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.yml"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.yml"
    )
    training_data = training_data.merge(training_data_responses)

    nlu_config = RasaNLUModelConfig({"language": "en", "pipeline": pipeline})

    trainer = Trainer(nlu_config)
    trainer.train(training_data)

    persisted_path = trainer.persist(tmpdir)

    assert trainer.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)
    parsed = loaded.parse("hello")
    assert loaded.pipeline
    assert parsed is not None
    assert (parsed.get("response_selector").get("all_retrieval_intents")) == [
        "chitchat"
    ]
    assert (
        parsed.get("response_selector")
        .get("default")
        .get("response")
        .get("intent_response_key")
    ) is not None
    assert (
        parsed.get("response_selector")
        .get("default")
        .get("response")
        .get("utter_action")
    ) is not None
    assert (
        parsed.get("response_selector").get("default").get("response").get("responses")
    ) is not None

    ranking = parsed.get("response_selector").get("default").get("ranking")
    assert ranking is not None

    for rank in ranking:
        assert rank.get("confidence") is not None
        assert rank.get("intent_response_key") is not None


def test_preprocess_selector_multiple_retrieval_intents():

    # use some available data
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.yml"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.yml"
    )
    training_data_extra_intent = TrainingData(
        [
            Message.build(
                text="Is it possible to detect the version?", intent="faq/q1"
            ),
            Message.build(text="How can I get a new virtual env", intent="faq/q2"),
        ]
    )
    training_data = training_data.merge(training_data_responses).merge(
        training_data_extra_intent
    )

    response_selector = ResponseSelector()

    response_selector.preprocess_train_data(training_data)

    assert sorted(response_selector.all_retrieval_intents) == ["chitchat", "faq"]


@pytest.mark.parametrize(
    "use_text_as_label, label_values",
    [
        [False, ["chitchat/ask_name", "chitchat/ask_weather"]],
        [True, ["I am Mr. Bot", "It's sunny where I live"]],
    ],
)
def test_preprocess_creates_index_label_mapping(use_text_as_label, label_values):

    # use data that include some responses
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.yml"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.yml"
    )
    training_data = training_data.merge(training_data_responses)
    FeatureGenerator.add_dense_dummy_features_to_messages(training_data.intent_examples)

    response_selector = ResponseSelector(
        component_config={"use_text_as_label": use_text_as_label}
    )
    response_selector.preprocess_train_data(training_data)

    assert response_selector.responses == training_data.responses
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
def test_preprocess_resolve_intent_response_key_from_label(
    predicted_label, train_on_text, resolved_intent_response_key
):

    # use data that include some responses
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.yml"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.yml"
    )
    training_data = training_data.merge(training_data_responses)

    # add dummy features
    FeatureGenerator.add_dense_dummy_features_to_messages(training_data)

    response_selector = ResponseSelector(
        component_config={"use_text_as_label": train_on_text}
    )
    response_selector.preprocess_train_data(training_data)

    label_intent_response_key = response_selector._resolve_intent_response_key(
        {"id": hash(predicted_label), "name": predicted_label}
    )
    assert resolved_intent_response_key == label_intent_response_key
    assert (
        response_selector.responses[
            util.intent_response_key_to_template_key(label_intent_response_key)
        ]
        == training_data.responses[
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


async def _train_persist_load_with_different_settings(
    pipeline: List[Dict[Text, Any]],
    component_builder: ComponentBuilder,
    tmp_path: Path,
    should_finetune: bool,
):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trainer, trained, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data="data/examples/rasa/demo-rasa.yml",
        component_builder=component_builder,
    )

    assert trainer.pipeline
    assert trained.pipeline

    loaded = Interpreter.load(
        persisted_path,
        component_builder,
        new_config=_config if should_finetune else None,
    )

    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") == trained.parse("Rasa is great!")


@pytest.mark.skip_on_windows
async def test_train_persist_load(component_builder: ComponentBuilder, tmpdir: Path):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {"name": "ResponseSelector", EPOCHS: 1},
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, True
    )


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
    component_builder: ComponentBuilder,
    tmp_path: Path,
    classifier_params: Dict[Text, Any],
    output_length: int,
    monkeypatch: MonkeyPatch,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "ResponseSelector"
    )
    assert pipeline[2]["name"] == "ResponseSelector"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data="data/test_selectors",
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    parse_data = loaded.parse("hello")
    response_ranking = parse_data.get("response_selector").get("default").get("ranking")

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
    component_builder: ComponentBuilder,
    tmp_path: Path,
    classifier_params: Dict[Text, int],
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "ResponseSelector"
    )
    assert pipeline[2]["name"] == "ResponseSelector"
    pipeline[2].update(classifier_params)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data="data/test_selectors",
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    response_ranking = parse_data.get("response_selector").get("default").get("ranking")

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
    component_builder: ComponentBuilder,
    tmp_path: Path,
    classifier_params: Dict[Text, int],
    data_path: Text,
    output_length: int,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "ResponseSelector"
    )
    assert pipeline[2]["name"] == "ResponseSelector"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data=data_path,
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    response_ranking = parse_data.get("response_selector").get("default").get("ranking")
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
    config: Dict[Text, Union[int, Dict[Text, List[int]]]], should_raise_warning: bool
):
    """ResponseSelector recommends disabling hidden layers if transformer is enabled."""
    with pytest.warns(UserWarning) as records:
        _ = ResponseSelector(component_config=config)
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
    config: Dict[Text, Optional[int]], should_set_default_transformer_size: bool,
):
    """ResponseSelector ensures sensible transformer size when transformer enabled."""
    default_transformer_size = 256
    with pytest.warns(UserWarning) as records:
        selector = ResponseSelector(component_config=config)

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


class SimpleSelectorTestCase(FeatureGenerator):
    def generate_input_and_expected_extracted_features(
        self, features_for_label_given: bool, label: Text,
    ) -> Dict[Text, Any]:

        # Re-use the test case from the intent prediction - as messages without
        # responses and without intent response keys ...
        other_test_case = SimpleIntentClassificationTestCaseWithEntities(
            featurizer_descriptions=self.featurizer_descriptions,
            used_featurizers=self.used_featurizers,
        )
        (
            messages_not_used_for_training,
            _,
        ) = other_test_case.generate_input_and_concatenated_features(
            input_contains_features_for_intent=False,
        )
        # ... but drop the features created for the other test case:
        for message in messages_not_used_for_training:
            message.features = []

        # TODO: we could append more messages here that would not be used for
        # training, e.g. without text but a response (because "core" message)

        # Create messages with intent response keys and/or text
        # Note: we prepend "A", "B", "C" to test the lexicographical sorting
        # that is supposed to happen.
        response_key_to_text = {
            "intent1/C-faq": "C Just one response.",
            "intent2/B-bielefeld_exists": "B does bielfeld exist?",
            "intent2/A-bielefeld_does_not": "A bielefeld doesn't exist, does it?",
        }
        messages_with_responses = []
        for i in range(3):
            # ... 3 of each kind:
            # - with intent_response_key and response
            # - with intent_response_key only
            # - with response only
            for intent_response_key, response in response_key_to_text.items():
                new_message = Message()
                if i in [0, 1]:
                    new_message.set(INTENT_RESPONSE_KEY, intent_response_key)
                if i in [0, 2]:
                    new_message.set(RESPONSE, response)
                new_message.set(INTENT, intent_response_key.split("/")[0])
                new_message.set(TEXT, "non-empty-text")
                # this *is* important because messages without
                new_message.set("other", "an-irrelevant-attribute")
                messages_with_responses.append(new_message)

        # Add messages and generate expected outputs
        all_messages = messages_not_used_for_training + messages_with_responses

        # Add some features to all messages
        attributes_with_features = [TEXT]
        if features_for_label_given:
            attributes_with_features += [label]
        concatenated_features = self.generate_features_for_all_messages(
            all_messages, add_to_messages=True, attributes=attributes_with_features,
        )

        # Add features for some unrelated attributes to the messages
        # (to illustrate that this attribute will not show up in the final model data)
        other = ["other"] + ([INTENT_RESPONSE_KEY] if label == RESPONSE else [RESPONSE])
        _ = self.generate_features_for_all_messages(
            all_messages, add_to_messages=True, attributes=other,
        )

        return all_messages, concatenated_features


@pytest.mark.parametrize(
    (
        "use_text_as_label, retrieval_intent, input_contains_features_for_label, "
        "used_featurizers, num_transformer_layers"
    ),
    itertools.product(
        [True, False],
        ["intent1", "intent2", None],
        [True, False],
        [["1", "2"], None],
        [0, 1],
    ),
)
def test_preprocess_train_data_and_create_data_for_prediction(
    use_text_as_label: bool,
    retrieval_intent: Text,
    input_contains_features_for_label: bool,
    used_featurizers: Optional[List[Text]],
    num_transformer_layers: int,
):

    # Describe and choose featurizers
    featurizer_descriptions = [
        FeaturizerDescription(
            name="1", seq_feat_dim=2, sent_feat_dim=2, sparse=True, dense=False
        ),
        FeaturizerDescription(
            name="2", seq_feat_dim=4, sent_feat_dim=4, sparse=False, dense=False
        ),
        FeaturizerDescription(
            name="3", seq_feat_dim=8, sent_feat_dim=8, sparse=True, dense=True
        ),
    ]
    used_featurizers = used_featurizers or ["1", "2", "3"]
    assert used_featurizers is None or set(used_featurizers) <= {"1", "2", "3"}

    # Use the following config ...
    config = {
        USE_TEXT_AS_LABEL: use_text_as_label,
        RETRIEVAL_INTENT: retrieval_intent,
        NUM_TRANSFORMER_LAYERS: num_transformer_layers,
        FEATURIZERS: used_featurizers,
    }
    # ... with these additional settings (to avoid warnings) ...
    if num_transformer_layers:
        config[TRANSFORMER_SIZE] = 42
        config[HIDDEN_LAYERS_SIZES] = {"text": [], "label": []}
    config[CONSTRAIN_SIMILARITIES] = True
    config[EVAL_NUM_EPOCHS] = 1
    # ... to instantiate the model:
    model = ResponseSelector(config)

    # In contrast to DIET, ....
    # (1) the label is never None here
    assert model.label_attribute in [RESPONSE, INTENT_RESPONSE_KEY]
    # (2) sometimes we skip sequence features for "TEXT"
    assert model._uses_sequence_features_for_input_text() == (
        num_transformer_layers > 0 or not use_text_as_label
    )

    # Create test data
    test_case = SimpleSelectorTestCase(
        featurizer_descriptions=featurizer_descriptions,
        used_featurizers=used_featurizers,
    )
    (
        all_messages,
        expected_outputs,
    ) = test_case.generate_input_and_expected_extracted_features(
        features_for_label_given=input_contains_features_for_label,
        label=model.label_attribute,
    )

    # Preprocess Training Data
    messages_for_training = copy.deepcopy(all_messages)
    model_data_for_training = model.preprocess_train_data(
        training_data=TrainingData(messages_for_training)
    )

    # Check index to label id mapping
    response_A = "A bielefeld doesn't exist, does it?"
    key_2A = "intent2/A-bielefeld_does_not"
    response_B = "B does bielfeld exist?"
    key_2B = "intent2/B-bielefeld_exists"
    response_C = "C Just one response."
    key_1C = "intent1/C-faq"
    idx_to_label = model.index_label_id_mapping
    if retrieval_intent == "intent1":
        if model.label_attribute == RESPONSE:
            assert idx_to_label == {0: response_C}
        else:
            assert idx_to_label == {0: key_1C}
    elif retrieval_intent == "intent2":
        if model.label_attribute == RESPONSE:
            assert idx_to_label == {0: response_A, 1: response_B}
        else:
            assert idx_to_label == {0: key_2A, 1: key_2B}
    else:
        if model.label_attribute == RESPONSE:
            assert idx_to_label == {0: response_A, 1: response_B, 2: response_C}
        else:
            assert idx_to_label == {0: key_1C, 1: key_2A, 2: key_2B}

    # Imitate Creation of Model Data during Predict
    messages_for_prediction = copy.deepcopy(all_messages)
    model_data_for_prediction = model._create_model_data(
        messages=messages_for_prediction,
        training=False,
        label_id_dict=model.index_label_id_mapping,
    )

    # Compare with expected results
    for model_data, training in [
        (model_data_for_training, True),
        (model_data_for_prediction, False),
    ]:

        # Response Selector will filter the training data...
        if training:
            if retrieval_intent is not None:
                relevant_indices = [
                    idx
                    for idx, message in enumerate(all_messages)
                    if message.get(INTENT) == retrieval_intent
                    and model.label_attribute in message.data
                ]
                # Observe that the following works for both possible labels
                # (response and intent_response_key)
                if retrieval_intent == "intent1":
                    assert len(relevant_indices) == 2
                else:
                    assert len(relevant_indices) == 4

            else:
                relevant_indices = [
                    idx
                    for idx, message in enumerate(all_messages)
                    if model.label_attribute in message.data
                ]
                assert len(relevant_indices) == 6  # = 2 + 4
        else:
            relevant_indices = list(range(len(all_messages)))

        # Check size and attributes
        assert model_data.number_of_examples() == len(relevant_indices)
        expected_keys = {TEXT}
        if training:
            expected_keys.add(LABEL)
        assert set(model_data.keys()) == expected_keys

        # Check subkeys for TEXT
        if model._uses_sequence_features_for_input_text():
            expected_text_sub_keys = {MASK, SENTENCE, SEQUENCE, SEQUENCE_LENGTH}
        else:
            expected_text_sub_keys = {MASK, SENTENCE}
        assert set(model_data.get(TEXT).keys()) == expected_text_sub_keys
        text_features = model_data.get(TEXT)
        # - subkey: mask (this is a "turn" mask)
        mask_features = text_features.get(MASK)
        assert len(mask_features) == 1
        mask = np.array(mask_features[0])  # because it's a feature array
        assert mask.shape == (len(relevant_indices), 1, 1)
        assert np.all(mask == 1)
        # - subkey: sequence-length
        if model._uses_sequence_features_for_input_text():
            length_features = text_features.get(SEQUENCE_LENGTH)
            assert len(length_features) == 1
            lengths = np.array(length_features[0])  # because it's a feature array
            assert lengths.shape == (len(relevant_indices),)
            if training:
                assert np.all(lengths == test_case.seq_len)
            else:
                expected_lengths = [
                    test_case.seq_len * bool(message.get(TEXT))
                    for message in all_messages
                ]
                assert np.all(lengths == expected_lengths)
        # - subkey: sentence and, if used, sequence
        if not model._uses_sequence_features_for_input_text():
            # Note that the comparison allow will assert that the actual features
            # do not contain keys that are not contained in the expected features:
            expected_outputs[TEXT].pop(FEATURE_TYPE_SEQUENCE, None)

        test_case.compare_features_of_same_type_and_sparseness(
            actual=text_features,
            expected=expected_outputs[TEXT],
            indices_of_expected=relevant_indices,
        )

        if training:

            # We already tested this, but we want to emphasize that:
            other_label = (
                INTENT_RESPONSE_KEY if model.label_attribute == RESPONSE else RESPONSE
            )
            assert other_label not in model_data.keys()

            # Check subkeys for LABEL
            expected_label_subkeys = {LABEL_SUB_KEY, MASK, SENTENCE}
            if input_contains_features_for_label:
                expected_label_subkeys.update({SEQUENCE, SEQUENCE_LENGTH})
            label_features = model_data.get(LABEL_KEY)
            assert set(label_features.keys()) == expected_label_subkeys
            # - subkey: ids
            # Note that the indices are sorted in the following because we
            # enforced the lexicographical sorting above.
            if retrieval_intent == "intent1":
                # "just one response"
                expected_ids = [0, 0]
            elif retrieval_intent == "intent2":
                # two responses
                expected_ids = [1, 0, 1, 0]
            else:
                if model.label_attribute == RESPONSE:
                    # because responses are sorted lexicographically (decreasing)
                    expected_ids = [2, 1, 0, 2, 1, 0]
                else:
                    # because intent1/C... comes before all intent2/[A|B]...
                    expected_ids = [0, 2, 1, 0, 2, 1]
            id_features = label_features.get(IDS)
            assert len(id_features) == 1
            ids = np.array(id_features[0])
            assert ids.shape == (len(relevant_indices), 1)
            assert np.all(ids.flatten() == expected_ids)
            # - subkey: mask (this is a "turn" mask)
            mask_features = label_features.get(MASK)
            assert len(mask_features) == 1
            mask = np.array(mask_features[0])
            assert mask.shape == (len(relevant_indices), 1, 1)
            assert np.all(mask == 1)

            if not input_contains_features_for_label:

                # we create features for LABEL on the fly if and only if no
                # features for labels are contained in the given messages
                assert SEQUENCE not in label_features
                assert len(label_features.get(SENTENCE)) == 1
                generated_label_features = np.array(label_features.get(SENTENCE)[0])
                if retrieval_intent == "intent1":
                    assert np.all(generated_label_features.flatten() == [1, 1])
                elif retrieval_intent == "intent2":
                    onehot_A = [[1.0, 0.0]]
                    onehot_B = [[0.0, 1.0]]
                    assert np.all(generated_label_features == [onehot_B, onehot_A] * 2)
                else:
                    if model.label_attribute == RESPONSE:
                        onehot_2A = [[1.0, 0.0, 0.0]]
                        onehot_2B = [[0.0, 1.0, 0.0]]
                        onehot_1C = [[0.0, 0.0, 1.0]]
                    else:
                        onehot_2A = [[0.0, 1.0, 0.0]]
                        onehot_2B = [[0.0, 0.0, 1.0]]
                        onehot_1C = [[1.0, 0.0, 0.0]]
                    assert np.all(
                        generated_label_features
                        == [onehot_1C, onehot_2B, onehot_2A] * 2
                    )

            else:
                # - subkey: sequence-length
                if model._uses_sequence_features_for_input_text():
                    length_features = label_features.get(SEQUENCE_LENGTH)
                    assert len(length_features) == 1
                    lengths = np.array(length_features[0])
                    assert lengths.shape == (len(relevant_indices),)
                    assert np.all(lengths == test_case.seq_len)
                # - subkey: sentence and sentence
                test_case.compare_features_of_same_type_and_sparseness(
                    actual=label_features,
                    expected=expected_outputs[model.label_attribute],
                    indices_of_expected=relevant_indices,
                )
