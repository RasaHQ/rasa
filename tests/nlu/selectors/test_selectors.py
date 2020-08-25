import pytest

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import load_data
from rasa.nlu.train import Trainer, Interpreter
from rasa.utils.tensorflow.constants import (
    EPOCHS,
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    TRAIN_ON_TEXT,
)
from rasa.nlu.constants import RESPONSE_SELECTOR_PROPERTY_NAME
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
    td = load_data("data/examples/rasa/demo-rasa.md")
    td_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
    td = td.merge(td_responses)

    nlu_config = RasaNLUModelConfig({"language": "en", "pipeline": pipeline})

    trainer = Trainer(nlu_config)
    trainer.train(td)

    persisted_path = trainer.persist(tmpdir)

    assert trainer.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)
    parsed = loaded.parse("hello")

    assert loaded.pipeline
    assert parsed is not None
    assert (
        parsed.get(RESPONSE_SELECTOR_PROPERTY_NAME)
        .get("default")
        .get("response")
        .get("full_retrieval_intent")
    ) is not None

    ranking = parsed.get(RESPONSE_SELECTOR_PROPERTY_NAME).get("default").get("ranking")
    assert ranking is not None

    for rank in ranking:
        assert rank.get("name") is not None
        assert rank.get("confidence") is not None
        assert rank.get("full_retrieval_intent") is not None


def test_training_label():

    # use data that include some responses
    td = load_data("data/examples/rasa/demo-rasa.md")
    td_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
    td = td.merge(td_responses)

    rs = ResponseSelector(component_config={TRAIN_ON_TEXT: False})
    rs.preprocess_train_data(td)

    assert rs.responses == td.responses
    assert sorted(list(rs.index_label_id_mapping.values())) == sorted(
        list(td.responses.keys())
    )

    rs = ResponseSelector(component_config={TRAIN_ON_TEXT: True})
    rs.preprocess_train_data(td)

    assert rs.responses == td.responses
    assert sorted(list(rs.index_label_id_mapping.values())) == sorted(
        [r[0].get("text") for r in td.responses.values()]
    )


@pytest.mark.parametrize(
    "pred_label, train_on_text, full_response_label, full_response_text",
    [
        ["chitchat/ask_name", False, "chitchat/ask_name", "I am Mr. Bot"],
        ["faq/ask_name", False, "faq/ask_name", "faq/ask_name"],
        ["faq/ask_name", True, "faq/ask_name", "faq/ask_name"],
        [
            "It's sunny where I live",
            True,
            "chitchat/ask_weather",
            "It's sunny where I live",
        ],
    ],
)
def test_resolve_responses(
    pred_label, train_on_text, full_response_label, full_response_text
):

    # use data that include some responses
    td = load_data("data/examples/rasa/demo-rasa.md")
    td_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
    td = td.merge(td_responses)

    rs = ResponseSelector(component_config={TRAIN_ON_TEXT: train_on_text})
    rs.preprocess_train_data(td)

    label_key, retrieved_response_label = rs._full_response(
        {"id": hash(pred_label), "name": pred_label}
    )
    assert label_key == full_response_label
    assert retrieved_response_label[0].get("text") == full_response_text
