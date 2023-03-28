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

    assert mapper.synonyms.get("mexican") is None
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
