from typing import Optional, Text

import pytest

import spacy.tokens.doc

from rasa.nlu.constants import DENSE_FEATURIZABLE_ATTRIBUTES, SPACY_DOCS
from rasa.nlu.model import InvalidModelError
from rasa.nlu.utils.spacy_utils import SpacyNLP, SpacyModel
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import ACTION_TEXT, RESPONSE, TEXT
from rasa.shared.nlu.training_data.message import Message


def create_spacy_nlp_component(
    model_name: Text = "en_core_web_md", case_sensitive: Optional[bool] = None
) -> SpacyNLP:
    component = SpacyNLP.create(
        {"model": model_name, "case_sensitive": case_sensitive}, None, None, None
    )
    return component


@pytest.mark.parametrize(
    "model_name,msg",
    [
        (
            "dinosaurhead",
            "Please confirm that dinosaurhead is an available spaCy model",
        ),
        (None, "Missing model configuration for `SpacyNLP` in `config.yml`"),
    ],
)
def test_model_raises_error_not_exist(model_name, msg):
    """It should throw a direct error when a bad model setting goes in."""
    with pytest.raises(InvalidModelError) as err:
        create_spacy_nlp_component(model_name)
    assert msg in str(err.value)


def test_spacy_spacy_model_provider():
    provider_component = create_spacy_nlp_component()
    spacy_model = provider_component.provide()
    assert spacy_model.model
    assert spacy_model.model_name == "en_core_web_md"


@pytest.mark.parametrize("case_sensitive", [True, False])
def test_spacy_preprocessor_adds_attributes_when_processing(
    case_sensitive: bool, spacy_model: SpacyModel
):
    preprocessor = create_spacy_nlp_component(case_sensitive=case_sensitive)
    message_data = {
        TEXT: "Hello my name is Joe",
        RESPONSE: "Some response",
        ACTION_TEXT: "Action Text",
    }
    message = Message(data=message_data)
    preprocessor.process([message], spacy_model)

    for attr, text in message_data.items():
        doc = message.data[SPACY_DOCS[attr]]

        assert isinstance(doc, spacy.tokens.doc.Doc)

        if case_sensitive:
            assert doc.text == text
        else:
            assert doc.text == text.lower()


def test_spacy_preprocessor_process_training_data(
    spacy_nlp_component: SpacyNLP, spacy_model: SpacyModel
):
    training_data = TrainingDataImporter.load_from_dict(
        training_data_paths=[
            "data/test_e2ebot/data/nlu.yml",
            "data/test_e2ebot/data/stories.yml",
        ]
    ).get_nlu_data()

    spacy_nlp_component.process_training_data(training_data, spacy_model)

    for message in training_data.training_examples:
        for attr in DENSE_FEATURIZABLE_ATTRIBUTES:
            attr_text = message.data.get(attr)
            if attr_text:
                doc = message.data[SPACY_DOCS[attr]]
                assert isinstance(doc, spacy.tokens.doc.Doc)
                assert doc.text == attr_text.lower()
