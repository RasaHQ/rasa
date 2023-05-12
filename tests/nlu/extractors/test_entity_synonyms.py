from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.shared.nlu.constants import TEXT, ENTITIES
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource


def test_entity_synonyms(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    resource = Resource("xy")
    entities = [
        {"entity": "test", "value": "chines", "start": 0, "end": 6},
        {"entity": "test", "value": "chinese", "start": 0, "end": 6},
        {"entity": "test", "value": "china", "start": 0, "end": 6},
    ]
    ent_synonyms = {"chines": "chinese", "NYC": "New York City"}

    mapper = EntitySynonymMapper.create(
        {}, default_model_storage, resource, default_execution_context, ent_synonyms
    )
    mapper.replace_synonyms(entities)

    assert len(entities) == 3
    assert entities[0]["value"] == "chinese"
    assert entities[1]["value"] == "chinese"
    assert entities[2]["value"] == "china"


def test_unintentional_synonyms_capitalized(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    resource = Resource("xy")
    mapper = EntitySynonymMapper.create(
        {}, default_model_storage, resource, default_execution_context
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

    mapper.train(TrainingData(training_examples=examples))

    assert mapper.synonyms.get("mexican") == "Mexican"
    assert mapper.synonyms.get("tacos") == "Mexican"


def test_synonym_mapper_with_ints(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    resource = Resource("xy")
    mapper = EntitySynonymMapper.create(
        {}, default_model_storage, resource, default_execution_context
    )
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
    mapper.process([message])

    assert message.get(ENTITIES) == entities


def test_synonym_alternate_case(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    resource = Resource("xy")
    mapper = EntitySynonymMapper.create(
        {}, default_model_storage, resource, default_execution_context
    )

    examples = [
        Message(
            data={
                TEXT: "What's the weather in austria?",
                "intent": "whats_weather",
                "entities": [
                    {"start": 22, "end": 29, "value": "austria", "entity": "GPE"}
                ],
            }
        ),
        Message(
            data={
                TEXT: "weather vienna?",
                "intent": "whats_weather",
                "entities": [
                    {"start": 8, "end": 14, "value": "austria", "entity": "GPE"}
                ],
            }
        ),
    ]
    entities = [
        {"entity": "test", "value": "austria", "start": 0, "end": 7},
        {"entity": "test", "value": "Austria", "start": 0, "end": 7},
        {"entity": "test", "value": "AUSTRIA", "start": 0, "end": 7},
        {"entity": "test", "value": "ausTRIA", "start": 0, "end": 7},
        {"entity": "test", "value": "Vienna", "start": 0, "end": 7},
        {"entity": "test", "value": "brazil", "start": 0, "end": 7},
    ]

    mapper.train(TrainingData(training_examples=examples))
    mapper.replace_synonyms(entities)
    expected_synonym_value = "austria"

    # synonym key for example value is present
    assert mapper.synonyms.get("vienna") == expected_synonym_value

    # synonym key for self is present
    assert mapper.synonyms.get("austria") == expected_synonym_value

    # all replacement values are correct
    assert entities[0]["value"] == expected_synonym_value
    assert entities[1]["value"] == expected_synonym_value
    assert entities[2]["value"] == expected_synonym_value
    assert entities[3]["value"] == expected_synonym_value
    assert entities[4]["value"] == expected_synonym_value
    assert entities[5]["value"] != expected_synonym_value
