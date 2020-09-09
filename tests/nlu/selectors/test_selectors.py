import pytest

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import load_data
from rasa.nlu.train import Trainer, Interpreter
from rasa.utils.tensorflow.constants import (
    EPOCHS,
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
)
from rasa.nlu.selectors.response_selector import ResponseSelector


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
    training_data = load_data("data/examples/rasa/demo-rasa.md")
    training_data_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
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
        .get("response_templates")
    ) is not None

    ranking = parsed.get("response_selector").get("default").get("ranking")
    assert ranking is not None

    for rank in ranking:
        assert rank.get("confidence") is not None
        assert rank.get("intent_response_key") is not None


@pytest.mark.parametrize(
    "use_text_as_label, label_values",
    [
        [False, ["chitchat/ask_name", "chitchat/ask_weather"]],
        [True, ["I am Mr. Bot", "It's sunny where I live"]],
    ],
)
def test_ground_truth_for_training(use_text_as_label, label_values):

    # use data that include some responses
    training_data = load_data("data/examples/rasa/demo-rasa.md")
    training_data_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
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
    training_data = load_data("data/examples/rasa/demo-rasa.md")
    training_data_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
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
        response_selector.responses[label_intent_response_key]
        == training_data.responses[resolved_intent_response_key]
    )
