from dataclasses import dataclass
import itertools
from pathlib import Path
import copy

import numpy as np
import pytest
from unittest.mock import Mock

from typing import List, Set, Text, Dict, Any, Tuple
from _pytest.monkeypatch import MonkeyPatch

import rasa.model
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.features import Features
import rasa.nlu.train
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.constants import (
    BILOU_ENTITIES,
    TOKENS_NAMES,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ENTITY_ATTRIBUTE_TYPE,
)
from rasa.utils.tensorflow.constants import (
    FEATURIZERS,
    LOSS_TYPE,
    MASK,
    RANDOM_SEED,
    RANKING_LENGTH,
    EPOCHS,
    MASKED_LM,
    SENTENCE,
    SEQUENCE,
    SEQUENCE_LENGTH,
    TENSORBOARD_LOG_LEVEL,
    TENSORBOARD_LOG_DIR,
    EVAL_NUM_EPOCHS,
    EVAL_NUM_EXAMPLES,
    CONSTRAIN_SIMILARITIES,
    CHECKPOINT_MODEL,
    BILOU_FLAG,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
    MODEL_CONFIDENCE,
    LINEAR_NORM,
)
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.model import Interpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils import train_utils
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.nlu.training_data.loading import load_data
from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.model_data_utils import FeatureArray
from rasa.utils.tensorflow.models import LABEL_KEY, LABEL_SUB_KEY
from tests.nlu.dummy_data.dummy_features import (
    ConcatenatedFeaturizations,
    DummyFeatures,
    FeaturizerDescription,
)
from tests.nlu.dummy_data.dummy_nlu_data import (
    TextIntentAndEntitiesDummy,
    IntentAndEntitiesEncodings,
)


def test_compute_default_label_features():
    label_features = [
        Message(data={TEXT: "test a"}),
        Message(data={TEXT: "test b"}),
        Message(data={TEXT: "test c"}),
        Message(data={TEXT: "test d"}),
    ]

    output = DIETClassifier._compute_default_label_features(label_features)

    output = output[0]

    for i, o in enumerate(output):
        assert isinstance(o, np.ndarray)
        assert o[0][i] == 1
        assert o.shape == (1, len(label_features))


def get_checkpoint_dir_path(path: Path, model_dir: Path) -> Path:
    """
    Produce the path of the checkpoint directory for DIET.

    This is coupled to the persist method of DIET.

    Args:
        model_dir: the model directory
        path: the path passed to train for training output.

    """
    return path / model_dir / "checkpoints"


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                    ],
                ),
                Message(
                    data={TEXT: "test b"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                    ],
                ),
            ],
            True,
        ),
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, INTENT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, INTENT, "test"),
                    ],
                )
            ],
            False,
        ),
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(2), FEATURE_TYPE_SEQUENCE, INTENT, "test")
                    ],
                )
            ],
            False,
        ),
    ],
)
def test_check_labels_features_exist(messages, expected):
    attribute = TEXT
    classifier = DIETClassifier()
    assert classifier._check_labels_features_exist(messages, attribute) == expected


@pytest.mark.parametrize(
    "messages, entity_expected",
    [
        (
            [
                Message(
                    data={
                        TEXT: "test a",
                        INTENT: "intent a",
                        ENTITIES: [
                            {"start": 0, "end": 4, "value": "test", "entity": "test"}
                        ],
                    },
                ),
                Message(
                    data={
                        TEXT: "test b",
                        INTENT: "intent b",
                        ENTITIES: [
                            {"start": 0, "end": 4, "value": "test", "entity": "test"}
                        ],
                    },
                ),
            ],
            True,
        ),
        (
            [
                Message(data={TEXT: "test a", INTENT: "intent a"},),
                Message(data={TEXT: "test b", INTENT: "intent b"},),
            ],
            False,
        ),
    ],
)
def test_model_data_signature_with_entities(
    messages: List[Message], entity_expected: bool
):
    classifier = DIETClassifier({"BILOU_flag": False})
    training_data = TrainingData(messages)

    # create tokens for entity parsing inside DIET
    tokenizer = WhitespaceTokenizer()
    tokenizer.train(training_data)

    model_data = classifier.preprocess_train_data(training_data)
    entity_exists = "entities" in model_data.get_signature().keys()
    assert entity_exists == entity_expected


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
        data="data/examples/rasa/demo-rasa-multi-intent.yml",
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
@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_different_settings_non_windows(
    component_builder: ComponentBuilder, tmp_path: Path
):
    pipeline = [
        {
            "name": "WhitespaceTokenizer",
            "intent_tokenization_flag": True,
            "intent_split_symbol": "+",
        },
        {"name": "CountVectorsFeaturizer"},
        {"name": "DIETClassifier", MASKED_LM: True, EPOCHS: 1},
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmp_path, should_finetune=False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmp_path, should_finetune=True
    )


@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_different_settings(component_builder, tmpdir):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {"name": "DIETClassifier", LOSS_TYPE: "margin", EPOCHS: 1},
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=True
    )


@pytest.mark.timeout(210, func_only=True)
async def test_train_persist_load_with_only_entity_recognition(
    component_builder, tmpdir
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "DIETClassifier",
            ENTITY_RECOGNITION: True,
            INTENT_CLASSIFICATION: False,
            EPOCHS: 1,
        },
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=True
    )


@pytest.mark.timeout(120, func_only=True)
async def test_train_persist_load_with_only_intent_classification(
    component_builder, tmpdir
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "DIETClassifier",
            ENTITY_RECOGNITION: False,
            INTENT_CLASSIFICATION: True,
            EPOCHS: 1,
        },
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=True
    )


async def test_raise_error_on_incorrect_pipeline(
    component_builder, tmp_path: Path, nlu_as_json_path: Text
):
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "DIETClassifier", EPOCHS: 1},
            ],
            "language": "en",
        }
    )

    with pytest.raises(Exception) as e:
        await rasa.nlu.train.train(
            _config,
            path=str(tmp_path),
            data=nlu_as_json_path,
            component_builder=component_builder,
        )

    assert "'DIETClassifier' requires 'Featurizer'" in str(e.value)


def as_pipeline(*components):
    return [{"name": c} for c in components]


@pytest.mark.parametrize(
    "classifier_params, data_path, output_length, output_should_sum_to_1",
    [
        (
            {RANDOM_SEED: 42, EPOCHS: 1},
            "data/test/many_intents.yml",
            10,
            True,
        ),  # default config
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 0, EPOCHS: 1},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH,
            False,
        ),  # no normalization
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 3, EPOCHS: 1},
            "data/test/many_intents.yml",
            3,
            True,
        ),  # lower than default ranking_length
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 12, EPOCHS: 1},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH,
            False,
        ),  # higher than default ranking_length
        (
            {RANDOM_SEED: 42, EPOCHS: 1},
            "data/test_moodbot/data/nlu.yml",
            7,
            True,
        ),  # less intents than default ranking_length
    ],
)
async def test_softmax_normalization(
    component_builder,
    tmp_path,
    classifier_params,
    data_path: Text,
    output_length,
    output_should_sum_to_1,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
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
    intent_ranking = parse_data.get("intent_ranking")
    # check that the output was correctly truncated after normalization
    assert len(intent_ranking) == output_length

    # check whether normalization had the expected effect
    output_sums_to_1 = sum(
        [intent.get("confidence") for intent in intent_ranking]
    ) == pytest.approx(1)
    assert output_sums_to_1 == output_should_sum_to_1

    # check whether the normalization of rankings is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


@pytest.mark.parametrize(
    "classifier_params, data_path",
    [
        (
            {
                RANDOM_SEED: 42,
                EPOCHS: 1,
                MODEL_CONFIDENCE: LINEAR_NORM,
                RANKING_LENGTH: -1,
            },
            "data/test_moodbot/data/nlu.yml",
        ),
    ],
)
async def test_inner_linear_normalization(
    component_builder: ComponentBuilder,
    tmp_path: Path,
    classifier_params: Dict[Text, Any],
    data_path: Text,
    monkeypatch: MonkeyPatch,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data=data_path,
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")

    # check whether normalization had the expected effect
    output_sums_to_1 = sum(
        [intent.get("confidence") for intent in intent_ranking]
    ) == pytest.approx(1)
    assert output_sums_to_1

    # check whether the normalization of rankings is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]

    # normalize shouldn't have been called
    mock.normalize.assert_not_called()


@pytest.mark.parametrize(
    "classifier_params, output_length",
    [({LOSS_TYPE: "margin", RANDOM_SEED: 42, EPOCHS: 1}, LABEL_RANKING_LENGTH)],
)
async def test_margin_loss_is_not_normalized(
    monkeypatch, component_builder, tmpdir, classifier_params, output_length
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data="data/test/many_intents.yml",
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")

    # check that the output was not normalized
    mock.normalize.assert_not_called()

    # check that the output was correctly truncated
    assert len(intent_ranking) == output_length

    # make sure top ranking is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


@pytest.mark.timeout(120, func_only=True)
async def test_set_random_seed(component_builder, tmpdir, nlu_as_json_path: Text):
    """test if train result is the same for two runs of tf embedding"""

    # set fixed random seed
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    ENTITY_RECOGNITION: False,
                    RANDOM_SEED: 1,
                    EPOCHS: 1,
                },
            ],
            "language": "en",
        }
    )

    # first run
    (trained_a, _, persisted_path_a) = await rasa.nlu.train.train(
        _config,
        path=tmpdir.strpath + "_a",
        data=nlu_as_json_path,
        component_builder=component_builder,
    )
    # second run
    (trained_b, _, persisted_path_b) = await rasa.nlu.train.train(
        _config,
        path=tmpdir.strpath + "_b",
        data=nlu_as_json_path,
        component_builder=component_builder,
    )

    loaded_a = Interpreter.load(persisted_path_a, component_builder)
    loaded_b = Interpreter.load(persisted_path_b, component_builder)
    result_a = loaded_a.parse("hello")["intent"]["confidence"]
    result_b = loaded_b.parse("hello")["intent"]["confidence"]

    assert result_a == result_b


@pytest.mark.parametrize("log_level", ["epoch", "batch"])
async def test_train_tensorboard_logging(
    log_level: Text,
    component_builder: ComponentBuilder,
    tmpdir: Path,
    nlu_data_path: Text,
):
    tensorboard_log_dir = Path(tmpdir / "tensorboard")

    assert not tensorboard_log_dir.exists()

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
                    "name": "DIETClassifier",
                    EPOCHS: 1,
                    TENSORBOARD_LOG_LEVEL: log_level,
                    TENSORBOARD_LOG_DIR: str(tensorboard_log_dir),
                    MODEL_CONFIDENCE: "linear_norm",
                    CONSTRAIN_SIMILARITIES: True,
                    EVAL_NUM_EXAMPLES: 15,
                    EVAL_NUM_EPOCHS: 1,
                },
            ],
            "language": "en",
        }
    )

    await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data=nlu_data_path,
        component_builder=component_builder,
    )

    assert tensorboard_log_dir.exists()

    all_files = list(tensorboard_log_dir.rglob("*.*"))
    assert len(all_files) == 2


async def test_train_model_checkpointing(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_data_path: Text,
):
    model_name = "nlu-checkpointed-model"
    model_dir = tmp_path / model_name
    checkpoint_dir = get_checkpoint_dir_path(tmp_path, model_dir)
    assert not checkpoint_dir.is_dir()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    EPOCHS: 2,
                    EVAL_NUM_EPOCHS: 1,
                    EVAL_NUM_EXAMPLES: 10,
                    CHECKPOINT_MODEL: True,
                },
            ],
            "language": "en",
        }
    )

    await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data=nlu_data_path,
        component_builder=component_builder,
        fixed_model_name=model_name,
    )

    assert checkpoint_dir.is_dir()

    """
    Tricky to validate the *exact* number of files that should be there, however there
    must be at least the following:
        - metadata.json
        - checkpoint
        - component_1_CountVectorsFeaturizer (as per the pipeline above)
        - component_2_DIETClassifier files (more than 1 file)
    """
    all_files = list(model_dir.rglob("*.*"))
    assert len(all_files) > 4


async def test_train_model_not_checkpointing(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_data_path: Text,
):
    model_name = "nlu-not-checkpointed-model"
    checkpoint_dir = get_checkpoint_dir_path(tmp_path, tmp_path / model_name)
    assert not checkpoint_dir.is_dir()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {"name": "DIETClassifier", EPOCHS: 2, CHECKPOINT_MODEL: False},
            ],
            "language": "en",
        }
    )

    await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data=nlu_data_path,
        component_builder=component_builder,
        fixed_model_name=model_name,
    )

    assert not checkpoint_dir.is_dir()


async def test_train_fails_with_zero_eval_num_epochs(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_data_path: Text,
):
    model_name = "nlu-fail"
    checkpoint_dir = get_checkpoint_dir_path(tmp_path, tmp_path / model_name)
    assert not checkpoint_dir.is_dir()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    EPOCHS: 1,
                    CHECKPOINT_MODEL: True,
                    EVAL_NUM_EPOCHS: 0,
                    EVAL_NUM_EXAMPLES: 10,
                },
            ],
            "language": "en",
        }
    )
    with pytest.raises(InvalidConfigException):
        with pytest.warns(UserWarning) as warning:
            await rasa.nlu.train.train(
                _config,
                path=str(tmp_path),
                data=nlu_data_path,
                component_builder=component_builder,
                fixed_model_name=model_name,
            )
    assert not checkpoint_dir.is_dir()
    warn_text = (
        f"You have opted to save the best model, but the value of '{EVAL_NUM_EPOCHS}' "
        f"is not -1 or greater than 0. Training will fail."
    )
    assert len([w for w in warning if warn_text in str(w.message)]) == 1


async def test_doesnt_checkpoint_with_zero_eval_num_examples(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_data_path: Text,
):
    model_name = "nlu-fail-checkpoint"
    checkpoint_dir = get_checkpoint_dir_path(tmp_path, tmp_path / model_name)
    assert not checkpoint_dir.is_dir()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    EPOCHS: 2,
                    CHECKPOINT_MODEL: True,
                    EVAL_NUM_EXAMPLES: 0,
                    EVAL_NUM_EPOCHS: 1,
                },
            ],
            "language": "en",
        }
    )
    with pytest.warns(UserWarning) as warning:
        await rasa.nlu.train.train(
            _config,
            path=str(tmp_path),
            data=nlu_data_path,
            component_builder=component_builder,
            fixed_model_name=model_name,
        )

    assert not checkpoint_dir.is_dir()
    warn_text = (
        f"You have opted to save the best model, but the value of "
        f"'{EVAL_NUM_EXAMPLES}' is not greater than 0. No checkpoint model "
        f"will be saved."
    )
    assert len([w for w in warning if warn_text in str(w.message)]) == 1


@pytest.mark.parametrize(
    "classifier_params",
    [
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: False},
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: True},
    ],
)
@pytest.mark.timeout(300, func_only=True)
async def test_train_persist_load_with_composite_entities(
    classifier_params, component_builder, tmpdir
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trainer, trained, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=tmpdir.strpath,
        data="data/test/demo-rasa-composite-entities.yml",
        component_builder=component_builder,
    )

    assert trainer.pipeline
    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    text = "I am looking for an italian restaurant"
    assert loaded.parse(text) == trained.parse(text)


async def test_process_gives_diagnostic_data(
    response_selector_interpreter: Interpreter,
):
    """Tests if processing a message returns attention weights as numpy array."""
    interpreter = response_selector_interpreter
    message = Message(data={TEXT: "hello"})
    for component in interpreter.pipeline:
        component.process(message)

    diagnostic_data = message.get(DIAGNOSTIC_DATA)

    # The last component is DIETClassifier, which should add attention weights
    name = f"component_{len(interpreter.pipeline) - 2}_DIETClassifier"
    assert isinstance(diagnostic_data, dict)
    assert name in diagnostic_data
    assert "attention_weights" in diagnostic_data[name]
    assert isinstance(diagnostic_data[name].get("attention_weights"), np.ndarray)
    assert "text_transformed" in diagnostic_data[name]
    assert isinstance(diagnostic_data[name].get("text_transformed"), np.ndarray)


@pytest.mark.parametrize(
    "initial_sparse_feature_sizes, final_sparse_feature_sizes, label_attribute",
    [
        (
            {
                TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]},
                INTENT: {FEATURE_TYPE_SEQUENCE: [5], FEATURE_TYPE_SENTENCE: []},
            },
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            INTENT,
        ),
        (
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            INTENT,
        ),
    ],
)
def test_removing_label_sparse_feature_sizes(
    initial_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    final_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    label_attribute: Text,
):
    """Tests if label attribute is removed from sparse feature sizes collection."""
    sparse_feature_sizes = DIETClassifier._remove_label_sparse_feature_sizes(
        sparse_feature_sizes=initial_sparse_feature_sizes,
        label_attribute=label_attribute,
    )
    assert sparse_feature_sizes == final_sparse_feature_sizes


@pytest.mark.timeout(120)
async def test_adjusting_layers_incremental_training(
    component_builder: ComponentBuilder, tmpdir: Path
):
    """Tests adjusting sparse layers of `DIETClassifier` to increased sparse
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
        {"name": "DIETClassifier", EPOCHS: 1},
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
    initial_diet_layers = (
        trained.pipeline[-1]
        .model._tf_layers["sequence_layer.text"]
        ._tf_layers["feature_combining"]
    )
    initial_diet_sequence_layer = initial_diet_layers._tf_layers[
        "sparse_dense.sequence"
    ]._tf_layers["sparse_to_dense"]
    initial_diet_sentence_layer = initial_diet_layers._tf_layers[
        "sparse_dense.sentence"
    ]._tf_layers["sparse_to_dense"]

    initial_diet_sequence_size = initial_diet_sequence_layer.get_kernel().shape[0]
    initial_diet_sentence_size = initial_diet_sentence_layer.get_kernel().shape[0]
    assert initial_diet_sequence_size == sum(
        old_sparse_feature_sizes[FEATURE_TYPE_SEQUENCE]
    )
    assert initial_diet_sentence_size == sum(
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

    final_diet_layers = (
        trained.pipeline[-1]
        .model._tf_layers["sequence_layer.text"]
        ._tf_layers["feature_combining"]
    )
    final_diet_sequence_layer = final_diet_layers._tf_layers[
        "sparse_dense.sequence"
    ]._tf_layers["sparse_to_dense"]
    final_diet_sentence_layer = final_diet_layers._tf_layers[
        "sparse_dense.sentence"
    ]._tf_layers["sparse_to_dense"]

    final_diet_sequence_size = final_diet_sequence_layer.get_kernel().shape[0]
    final_diet_sentence_size = final_diet_sentence_layer.get_kernel().shape[0]
    assert final_diet_sequence_size == sum(
        new_sparse_feature_sizes[FEATURE_TYPE_SEQUENCE]
    )
    assert final_diet_sentence_size == sum(
        new_sparse_feature_sizes[FEATURE_TYPE_SENTENCE]
    )
    # check if the data signatures were correctly updated
    new_data_signature = trained.pipeline[-1].model.data_signature
    new_predict_data_signature = trained.pipeline[-1].model.predict_data_signature
    iter2_data = load_data(iter2_data_path)
    expected_sequence_lengths = len(iter2_data.training_examples)

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
        {"name": "DIETClassifier", EPOCHS: 1},
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


class TextIntentAndEntitiesTestRuns:
    """Describes all test cases using the `TextIntentAndEntitiesDummy`."""

    CONFIGURATIONS = [
        {
            "component_config": {
                INTENT_CLASSIFICATION: intent_classification,
                ENTITY_RECOGNITION: entity_recognition,
                FEATURIZERS: used_featurizers,
                BILOU_FLAG: bilou_tagging,
                EPOCHS: 1,
                CONSTRAIN_SIMILARITIES: True,
            },
            "input_contains_features_for_intent": featurize_intent,
            "attributes_without_sentence_features": skip_sentence_features,
        }
        for skip_sentence_features in [set(), {TEXT}, {INTENT}, {TEXT, INTENT}]
        for used_featurizers in [None, ["3", "1"]]  # switch order and skip one
        for (
            featurize_intent,
            intent_classification,
            entity_recognition,
            bilou_tagging,
        ) in itertools.product([True, False], repeat=4)
        # skip if there's no goal
        if (intent_classification or entity_recognition)
        # skip bilou tagging without entities enabled
        and not (bilou_tagging and not entity_recognition)
        # skip intent featurization tests if we don't do intent featurization
        and not (not intent_classification and featurize_intent)
        # skip when we want to skip sentence features for intents but there are none
        and not (INTENT in skip_sentence_features and not featurize_intent)
        # skip some more (no interesting combinations)
        and not (len(skip_sentence_features) > 0 and used_featurizers is None)
    ]

    @dataclass
    class TestRunData:
        # generator used to create dummy data and features
        meta: TextIntentAndEntitiesDummy
        # dummy data used for this test run
        messages: List[Message]
        # features created for each message as used by the intent classifier
        # (i.e. features of `used_featurizers` are concatenated per feature type and
        # sparseness property)
        concatenated_features: List[ConcatenatedFeaturizations]
        trained_model: DIETClassifier
        # Each test run prepares data for training, `label_data` and prediction.
        # We collect the results for each of these tasks separately, i.e. the
        # used messages (inputs), resulting rasa model data (output) and the
        # expected results (e.g. which indices from the input are to be used).
        sub_results: Dict[
            Text, Tuple[List[Message], RasaModelData, IntentAndEntitiesEncodings]
        ]

    @staticmethod
    def run(
        component_config: Dict[Text, Any],
        input_contains_features_for_intent: bool,
        attributes_without_sentence_features: Set[Text],
    ) -> TestRunData:
        """Creates a model and prepares `TextIntentAndEntitiesData` dummy data."""
        # Create dummy data
        dummy_data = TextIntentAndEntitiesDummy(
            featurize_intents=input_contains_features_for_intent,
            no_sentence_features=attributes_without_sentence_features,
        )
        original_messages = dummy_data.create_messages()
        dummy_data.featurize_messages(original_messages)
        concatenated_features = dummy_data.create_and_concatenate_features(
            messages=original_messages, used_featurizers=component_config[FEATURIZERS],
        )
        expected_results = dummy_data.intent_classifier_usage()

        # Create the model
        model = DIETClassifier(component_config=component_config)

        # Preprocess training data
        messages_for_training = copy.deepcopy(original_messages)
        model_data_for_training = model.preprocess_train_data(
            training_data=TrainingData(messages_for_training)
        )

        # Imitate creation of model data during inference time
        messages_for_prediction = copy.deepcopy(original_messages)
        model_data_for_prediction = model._create_model_data(
            messages_for_prediction, training=False,
        )

        return TextIntentAndEntitiesTestRuns.TestRunData(
            meta=dummy_data,
            messages=original_messages,
            concatenated_features=concatenated_features,
            trained_model=model,
            sub_results={
                "training": (
                    messages_for_training,
                    model_data_for_training,
                    expected_results["training"],
                ),
                "label_data": (
                    original_messages,
                    model._label_data,
                    expected_results["label_data"],
                ),
                "prediction": (
                    messages_for_prediction,
                    model_data_for_prediction,
                    expected_results["prediction"],
                ),
            },
        )


@pytest.mark.parametrize("config", TextIntentAndEntitiesTestRuns.CONFIGURATIONS)
def test_create_data_label_mapping(config: Dict[Text, Any]):
    # 1. Note that this mapping is needed by predict, so we test this separately.
    # 2. Observe that the current version *always* computes this mapping, even if
    #    intent classification is switched off.
    test_run = TextIntentAndEntitiesTestRuns.run(**config)
    _, _, expected_result = test_run.sub_results["label_data"]
    # Note: we strip spaces here
    expected = {
        intent_index: test_run.messages[message_index].get(INTENT).strip()
        for intent_index, message_index in enumerate(expected_result.indices)
    }
    assert test_run.trained_model.index_label_id_mapping == expected


@pytest.mark.parametrize("config", TextIntentAndEntitiesTestRuns.CONFIGURATIONS)
def test_create_data_subkey_text(config: Dict[Text, Any]):
    test_run = TextIntentAndEntitiesTestRuns.run(**config)

    for key in test_run.sub_results:

        used_messages, model_data, expected_results = test_run.sub_results[key]

        # no TEXT features
        if key == "label_data":
            assert TEXT not in model_data.keys()
            continue

        # subkeys for TEXT
        text_features = model_data.get(TEXT)
        expected_text_sub_keys = {MASK, SEQUENCE, SEQUENCE_LENGTH}
        if TEXT not in config["attributes_without_sentence_features"]:
            expected_text_sub_keys.add(SENTENCE)
        assert set(text_features.keys()) == expected_text_sub_keys

        # subkey: mask (this is a "turn" mask)
        mask_features_array = text_features.get(MASK)
        assert len(mask_features_array) == 1
        mask = np.array(mask_features_array[0])
        assert mask.shape == (len(expected_results.indices), 1, 1)
        assert np.all(
            mask.flatten()
            == [
                1.0 if used_messages[idx].get(TEXT, None) else 0.0
                for idx in expected_results.indices
            ]
        )

        # subkey: sequence-length
        length_features_array = text_features.get(SEQUENCE_LENGTH)
        assert len(length_features_array) == 1
        lengths = np.array(length_features_array[0])
        assert lengths.shape == (len(expected_results.indices),)
        expected_lengths = [
            len(used_messages[idx].get(TOKENS_NAMES[TEXT], []))
            for idx in expected_results.indices
        ]
        assert np.all(lengths == expected_lengths)

        # subkey: sequence and possibly sentence (see above)
        DummyFeatures.compare_with_feature_arrays(
            expected=[
                test_run.concatenated_features[TEXT][idx]
                for idx in expected_results.indices
            ],
            actual=text_features,
        )

        # sparse feature sizes
        used_featurizer_names = config["component_config"][FEATURIZERS]
        all_featurizers: List[FeaturizerDescription] = test_run.meta.featurizers
        sparse_used_featurizers = [
            d
            for d in all_featurizers
            if d.is_sparse
            and ((used_featurizer_names is None) or (d.name in used_featurizer_names))
        ]
        actual_sizes_by_type = model_data.sparse_feature_sizes[TEXT]
        assert np.all(
            actual_sizes_by_type[FEATURE_TYPE_SEQUENCE]
            == [d.sequence_dim for d in sparse_used_featurizers]
        )
        if SENTENCE in expected_text_sub_keys:
            assert np.all(
                actual_sizes_by_type[FEATURE_TYPE_SENTENCE]
                == [d.sentence_dim for d in sparse_used_featurizers]
            )


@pytest.mark.parametrize(
    "config",
    [
        config
        for config in TextIntentAndEntitiesTestRuns.CONFIGURATIONS
        if config["component_config"].get(INTENT_CLASSIFICATION)
    ],
)
def test_create_data_subkey_label(config: Dict[Text, Any]):
    test_run = TextIntentAndEntitiesTestRuns.run(**config)
    input_contains_features_for_intent = config["input_contains_features_for_intent"]

    for key in test_run.sub_results:

        used_messages, model_data, expected_results = test_run.sub_results[key]
        has_mask_features = key != "label_data"

        # no label data
        if key == "prediction":
            # ... even though there is intent information in the data
            assert any(INTENT in message.data for message in used_messages)
            assert LABEL_KEY not in model_data.keys()
            continue

        # subkeys for label
        expected_label_subkeys = {LABEL_SUB_KEY}
        if has_mask_features:
            expected_label_subkeys.add(MASK)
        if input_contains_features_for_intent:
            expected_label_subkeys.update({SEQUENCE, SEQUENCE_LENGTH})
            if INTENT not in config["attributes_without_sentence_features"]:
                expected_label_subkeys.add(SENTENCE)
        else:
            expected_label_subkeys.update({SENTENCE})
        label_features = model_data.get(LABEL_KEY)
        assert set(label_features.keys()) == expected_label_subkeys

        # subkey: mask (this is a "turn" mask, hence all masks are just "[1]")
        if has_mask_features:
            assert len(label_features.get(MASK)) == 1
            mask = np.array(label_features.get(MASK)[0])
            assert np.all(mask == [1] * len(expected_results.indices))

        # subkey: label_sub_key / id
        id_features_array = label_features.get(LABEL_SUB_KEY)
        assert len(id_features_array) == 1
        id_features = np.array(id_features_array[0])
        assert np.all(id_features == [[id] for id in expected_results.intents_ids])

        # - subkey: sentence
        if not input_contains_features_for_intent:
            # - subkey: sequence and sequence length
            assert SEQUENCE not in label_features
            assert SEQUENCE_LENGTH not in label_features
            # - subkey: sentence
            sentence_feature_array = label_features.get(SENTENCE)
            assert len(sentence_feature_array) == 1
            generated_label_features = np.array(sentence_feature_array[0])
            assert np.all(generated_label_features == expected_results.intents_mhot)
        else:
            # - subkey: sequence-length
            length_feature_array = label_features.get(SEQUENCE_LENGTH)
            assert len(length_feature_array) == 1
            lengths = np.array(length_feature_array[0])
            expected_lengths = [
                len(used_messages[idx].get(TOKENS_NAMES[INTENT], []))
                for idx in expected_results.indices
            ]
            assert np.all(lengths == expected_lengths)
            # - subkeys: sequence and possibly sentence (see above)
            DummyFeatures.compare_with_feature_arrays(
                expected=[
                    test_run.concatenated_features[INTENT][idx]
                    for idx in expected_results.indices
                ],
                actual=label_features,
            )


@pytest.mark.parametrize(
    "config",
    [
        config
        for config in TextIntentAndEntitiesTestRuns.CONFIGURATIONS
        if config["component_config"].get(ENTITY_RECOGNITION)
    ],
)
def test_create_data_subkey_entities(config: Dict[Text, Any]):
    test_run = TextIntentAndEntitiesTestRuns.run(**config)
    component_config = config["component_config"]

    for key in test_run.sub_results:
        used_messages, model_data, expected_results = test_run.sub_results[key]

        # no entities
        if key in ["label_data", "prediction"]:
            assert any(ENTITIES in message.data for message in used_messages)
            assert ENTITIES not in model_data.keys()
            continue

        # sub-keys for ENTITIES
        expected_entity_sub_keys = {ENTITY_ATTRIBUTE_TYPE, MASK}
        entity_features = model_data.get(ENTITIES)
        assert entity_features
        assert set(entity_features.keys()) == expected_entity_sub_keys
        for key in entity_features:
            assert len(entity_features.get(key)) == 1
            assert isinstance(entity_features.get(key)[0], FeatureArray)

        # subkey: mask  (this is a "turn" mask, hence all masks are just "[1]")
        mask = np.array(entity_features.get(MASK)[0])
        expected_mask = [1] * len(used_messages)
        assert np.all(mask == expected_mask)

        # subkey: entities
        entity_sub_features = entity_features[ENTITY_ATTRIBUTE_TYPE][0]
        assert len(entity_sub_features) == len(expected_results.indices)
        for idx, orig_idx in enumerate(expected_results.indices):
            if component_config.get(BILOU_FLAG):
                assert np.all(
                    np.array(entity_sub_features[idx]).flatten()
                    == expected_results.entities_bilou_ids[idx]
                )
                assert (
                    used_messages[orig_idx].get(BILOU_ENTITIES)
                    == expected_results.entities_bilou_tags[idx]
                )
            else:
                assert np.all(
                    np.array(entity_sub_features[idx]).flatten()
                    == expected_results.entities_ids[idx]
                )
                assert not used_messages[orig_idx].get(BILOU_ENTITIES)
