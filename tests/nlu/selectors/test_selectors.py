from pathlib import Path

import pytest
import numpy as np
from typing import List, Dict, Text, Any, Optional
from mock import Mock
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
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    CONSTRAIN_SIMILARITIES,
    CHECKPOINT_MODEL,
    MODEL_CONFIDENCE,
    RANDOM_SEED,
    RANKING_LENGTH,
    LOSS_TYPE,
    USE_TEXT_AS_LABEL,
)
from rasa.utils import train_utils
from rasa.shared.nlu.constants import TEXT
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from tests.nlu.classifiers.test_diet_classifier import as_pipeline


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
def test_ground_truth_for_training(use_text_as_label, label_values):

    # use data that include some responses
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.yml"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.yml"
    )
    training_data = training_data.merge(training_data_responses)

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
def test_resolve_intent_response_key_from_label(
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
    "num_transformer_layers, should_raise_warning",
    [(5, True), (0, False), (-1, False)],
)
def test_DIET2DIET_config_warning_transformer_with_hidden_layers(
    num_transformer_layers: int, should_raise_warning: bool
):
    """DIET2DIET recommends to disable hidden layers if transformer is enabled."""
    with pytest.warns(UserWarning) as records:
        _ = ResponseSelector(
            component_config={
                USE_TEXT_AS_LABEL: True,
                NUM_TRANSFORMER_LAYERS: num_transformer_layers,
            }
        )

    if should_raise_warning:
        assert len(records) > 0
        # Check all warnings since there may be multiple other warnings we don't care
        # about in this test case.
        assert any(
            "We recommend to disable the hidden layers when using a transformer"
            in record.message.args[0]
            for record in records
        )
    else:
        # Check all warnings since there may be multiple other warnings we don't care
        # about in this test case.
        assert not any(
            "We recommend to disable the hidden layers when using a transformer"
            in record.message.args[0]
            for record in records
        )


@pytest.mark.parametrize(
    "transformer_config, should_set_default_transformer_size",
    [
        ({}, True),
        ({TRANSFORMER_SIZE: 0}, True),
        ({TRANSFORMER_SIZE: -1}, True),
        ({TRANSFORMER_SIZE: None}, True),
        ({TRANSFORMER_SIZE: 10}, False),
    ],
)
def test_DIET2DIET_sets_transformer_size_when_needed(
    transformer_config: Dict[Text, Optional[int]],
    should_set_default_transformer_size: bool,
):
    """DIET2DIET ensures a sensible transformer size when transformer is enabled."""
    diet2diet_default_transformer_size = 256
    component_config = dict({USE_TEXT_AS_LABEL: True, NUM_TRANSFORMER_LAYERS: 5})
    component_config.update(transformer_config)
    with pytest.warns(UserWarning) as records:
        selector = ResponseSelector(component_config=component_config)
    if should_set_default_transformer_size:
        assert len(records) > 0
        assert any(
            f"a positive size is required when using `{USE_TEXT_AS_LABEL}="
            f"True` together with `{NUM_TRANSFORMER_LAYERS} > 0`"
            in record.message.args[0]
            for record in records
        )
        assert (
            selector.component_config[TRANSFORMER_SIZE]
            == diet2diet_default_transformer_size
        )
    else:
        assert not any(
            f"a positive size is required when using `{USE_TEXT_AS_LABEL}="
            f"True` together with `{NUM_TRANSFORMER_LAYERS} > 0`"
            in record.message.args[0]
            for record in records
        )
        assert (
            selector.component_config[TRANSFORMER_SIZE]
            == transformer_config[TRANSFORMER_SIZE]
        )
