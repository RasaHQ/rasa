from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message


def test_spacy_ner_extractor(component_builder, spacy_nlp):
    _config = RasaNLUModelConfig({"pipeline": [{"name": "SpacyEntityExtractor"}]})
    ext = component_builder.create_component(_config.for_component(0), _config)
    example = Message(
        "anywhere in the U.K.",
        {
            "intent": "restaurant_search",
            "entities": [],
            "text_spacy_doc": spacy_nlp("anywhere in the west"),
        },
    )

    ext.process(example, spacy_nlp=spacy_nlp)

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
        "anywhere in the West with Sebastian Thrun",
        {
            "intent": "example_intent",
            "entities": [],
            "text_spacy_doc": spacy_nlp("anywhere in the West with Sebastian Thrun"),
        },
    )
    _config = RasaNLUModelConfig({"pipeline": [{"name": "SpacyEntityExtractor"}]})

    _config.set_component_attr(0, dimensions=["PERSON"])
    ext = component_builder.create_component(_config.for_component(0), _config)
    ext.process(example, spacy_nlp=spacy_nlp)

    assert len(example.get("entities", [])) == 1
    assert example.get("entities")[0] == {
        "start": 26,
        "extractor": "SpacyEntityExtractor",
        "end": 41,
        "value": "Sebastian Thrun",
        "entity": "PERSON",
        "confidence": None,
    }
