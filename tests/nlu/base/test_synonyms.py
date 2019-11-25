from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.model import Metadata
import pytest


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
