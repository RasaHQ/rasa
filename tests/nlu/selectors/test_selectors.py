import pytest

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import load_data
from rasa.nlu.train import Trainer, Interpreter
from rasa.utils.tensorflow.constants import EPOCHS
from rasa.nlu.constants import RESPONSE_SELECTOR_PROPERTY_NAME


@pytest.mark.parametrize(
    "pipeline",
    [
        [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {"name": "ResponseSelector", EPOCHS: 1},
        ]
    ],
)
def test_train_selector(pipeline, component_builder, tmpdir):
    # use data that include some responses
    td = load_data("data/examples/rasa/demo-rasa.md")
    td_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
    td = td.merge(td_responses)
    td.fill_response_phrases()

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
        .get("full_retrieval_intent")
    ) is not None
