from rasa.nlu.training_data import TrainingData, Message
from tests.nlu import utilities


def test_unintentional_synonyms_capitalized(component_builder):
    _config = utilities.base_test_conf("pretrained_embeddings_spacy")
    ner_syn = component_builder.create_component(_config.for_component(5), _config)
    examples = [
        Message(
            "Any Mexican restaurant will do",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 4, "end": 11, "value": "Mexican", "entity": "cuisine"}
                ],
            },
        ),
        Message(
            "I want Tacos!",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 7, "end": 12, "value": "Mexican", "entity": "cuisine"}
                ],
            },
        ),
    ]
    ner_syn.train(TrainingData(training_examples=examples), _config)
    assert ner_syn.synonyms.get("mexican") is None
    assert ner_syn.synonyms.get("tacos") == "Mexican"
