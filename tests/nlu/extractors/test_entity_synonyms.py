from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapperComponent
from rasa.shared.nlu.constants import TEXT, ENTITIES
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message


def test_entity_synonyms():
    entities = [
        {"entity": "test", "value": "chines", "start": 0, "end": 6},
        {"entity": "test", "value": "chinese", "start": 0, "end": 6},
        {"entity": "test", "value": "china", "start": 0, "end": 6},
    ]
    ent_synonyms = {"chines": "chinese", "NYC": "New York City"}
    EntitySynonymMapperComponent(synonyms=ent_synonyms).replace_synonyms(entities)
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
            data={
                TEXT: "Any Mexican restaurant will do",
                "intent": "restaurant_search",
                "entities": [
                    {"start": 4, "end": 11, "value": "Mexican", "entity": "cuisine"}
                ],
            }
        ),
        Message(
            data={
                TEXT: "I want Tacos!",
                "intent": "restaurant_search",
                "entities": [
                    {"start": 7, "end": 12, "value": "Mexican", "entity": "cuisine"}
                ],
            }
        ),
    ]

    ner_syn.train(
        TrainingData(training_examples=examples), pretrained_embeddings_spacy_config
    )

    assert ner_syn.synonyms.get("mexican") is None
    assert ner_syn.synonyms.get("tacos") == "Mexican"


def test_synonym_mapper_with_ints():
    mapper = EntitySynonymMapperComponent()
    entities = [
        {
            "start": 21,
            "end": 22,
            "text": "5",
            "value": 5,
            "confidence": 1.0,
            "additional_info": {"value": 5, "type": "value"},
            "entity": "number",
            "extractor": "DucklingEntityExtractorComponent",
        }
    ]
    message = Message(data={TEXT: "He was 6 feet away", ENTITIES: entities})

    # This doesn't break
    mapper.process(message)

    assert message.get(ENTITIES) == entities
