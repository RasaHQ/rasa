import pytest

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import load_data
from rasa.nlu.train import Trainer, Interpreter
from rasa.utils.tensorflow.constants import EPOCHS


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

    assert loaded.pipeline
    assert loaded.parse("hello") is not None
