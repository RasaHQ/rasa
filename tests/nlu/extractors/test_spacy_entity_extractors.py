from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.nlu.utils.spacy_utils import SpacyModel
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message


def test_spacy_ner_extractor(spacy_nlp):
    example = Message(
        data={
            TEXT: "anywhere in the U.K.",
            "intent": "restaurant_search",
            "entities": [],
            "text_spacy_doc": spacy_nlp("anywhere in the west"),
        }
    )

    component = SpacyEntityExtractor(SpacyEntityExtractor.get_default_config())

    component.process([example], model=SpacyModel(model=spacy_nlp, model_name=""))

    assert len(example.get("entities", [])) == 1
    assert example.get("entities")[0] == {
        "start": 16,
        "extractor": "SpacyEntityExtractor",
        "end": 20,
        "value": "U.K.",
        "entity": "GPE",
        "confidence": None,
    }

    # Test dimension filtering includes only specified dimensions
    example = Message(
        data={
            TEXT: "anywhere in the West with Sebastian Thrun",
            "intent": "example_intent",
            "entities": [],
            "text_spacy_doc": spacy_nlp("anywhere in the West with Sebastian Thrun"),
        }
    )

    component = SpacyEntityExtractor({"dimensions": ["PERSON"]})
    component.process([example], model=SpacyModel(model=spacy_nlp, model_name=""))

    assert len(example.get("entities", [])) == 1
    assert example.get("entities")[0] == {
        "start": 26,
        "extractor": "SpacyEntityExtractor",
        "end": 41,
        "value": "Sebastian Thrun",
        "entity": "PERSON",
        "confidence": None,
    }
