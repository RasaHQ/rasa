from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.training_data import TrainingData, Message


def test_entity_synonyms():
    entities = [
        {"entity": "test", "value": "chines", "start": 0, "end": 6},
        {"entity": "test", "value": "chinese", "start": 0, "end": 6},
        {"entity": "test", "value": "china", "start": 0, "end": 6},
    ]
    ent_synonyms = {"chines": "chinese", "NYC": "New York City"}
    EntitySynonymMapper(synonyms=ent_synonyms).replace_synonyms(entities)
    assert len(entities) == 3
    assert entities[0]["value"] == "chinese"
    assert entities[1]["value"] == "chinese"
    assert entities[2]["value"] == "china"


def test_unintentional_synonyms_capitalized(
    component_builder, pretrained_embeddings_spacy_config
):
    idx = pretrained_embeddings_spacy_config.component_names.index(
        "EntitySynonymMapper"
    )
    ner_syn = component_builder.create_component(
        pretrained_embeddings_spacy_config.for_component(idx),
        pretrained_embeddings_spacy_config,
    )

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

    ner_syn.train(
        TrainingData(training_examples=examples), pretrained_embeddings_spacy_config
    )

    assert ner_syn.synonyms.get("mexican") is None
    assert ner_syn.synonyms.get("tacos") == "Mexican"
