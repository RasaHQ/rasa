from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import load_data
from rasa.nlu.train import Trainer, Interpreter
from rasa.utils.tensorflow.constants import EPOCHS


def test_train_response_selector(component_builder, tmpdir):
    td = load_data("data/examples/rasa/demo-rasa.md")
    td_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
    td = td.merge(td_responses)
    td.fill_response_phrases()

    nlu_config = RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {"name": "DIETClassifier", EPOCHS: 2},
                {"name": "ResponseSelector", EPOCHS: 2},
            ],
        }
    )

    trainer = Trainer(nlu_config)
    trainer.train(td)

    persisted_path = trainer.persist(tmpdir)

    assert trainer.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


def test_train_diet_selector(component_builder, tmpdir):
    td = load_data("data/examples/rasa/demo-rasa.md")
    td_responses = load_data("data/examples/rasa/demo-rasa-responses.md")
    td = td.merge(td_responses)
    td.fill_response_phrases()

    nlu_config = RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {"name": "DIETClassifier", EPOCHS: 2},
                {"name": "DIETSelector", EPOCHS: 2},
            ],
        }
    )

    trainer = Trainer(nlu_config)
    trainer.train(td)

    persisted_path = trainer.persist(tmpdir)

    assert trainer.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None
