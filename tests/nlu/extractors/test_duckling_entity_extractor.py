import responses

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.duckling_entity_extractor import (
    DucklingEntityExtractorComponent,
)
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message


def test_duckling_entity_extractor_with_multiple_extracted_dates(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    config = {
        **DucklingEntityExtractorComponent.get_default_config(),
        "dimensions": ["time"],
        "timezone": "UTC",
        "url": "http://localhost:8000",
    }
    duckling = DucklingEntityExtractorComponent.create(
        config, default_model_storage, Resource("duckling"), default_execution_context,
    )

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://localhost:8000/parse",
            json=[
                {
                    "body": "Today",
                    "start": 0,
                    "value": {
                        "values": [
                            {
                                "value": "2018-11-13T00:00:00.000-08:00",
                                "grain": "day",
                                "type": "value",
                            }
                        ],
                        "value": "2018-11-13T00:00:00.000-08:00",
                        "grain": "day",
                        "type": "value",
                    },
                    "end": 5,
                    "dim": "time",
                    "latent": False,
                },
                {
                    "body": "the 5th",
                    "start": 9,
                    "value": {
                        "values": [
                            {
                                "value": "2018-12-05T00:00:00.000-08:00",
                                "grain": "day",
                                "type": "value",
                            },
                            {
                                "value": "2019-01-05T00:00:00.000-08:00",
                                "grain": "day",
                                "type": "value",
                            },
                            {
                                "value": "2019-02-05T00:00:00.000-08:00",
                                "grain": "day",
                                "type": "value",
                            },
                        ],
                        "value": "2018-12-05T00:00:00.000-08:00",
                        "grain": "day",
                        "type": "value",
                    },
                    "end": 16,
                    "dim": "time",
                    "latent": False,
                },
                {
                    "body": "5th of May",
                    "start": 13,
                    "value": {
                        "values": [
                            {
                                "value": "2019-05-05T00:00:00.000-07:00",
                                "grain": "day",
                                "type": "value",
                            },
                            {
                                "value": "2020-05-05T00:00:00.000-07:00",
                                "grain": "day",
                                "type": "value",
                            },
                            {
                                "value": "2021-05-05T00:00:00.000-07:00",
                                "grain": "day",
                                "type": "value",
                            },
                        ],
                        "value": "2019-05-05T00:00:00.000-07:00",
                        "grain": "day",
                        "type": "value",
                    },
                    "end": 23,
                    "dim": "time",
                    "latent": False,
                },
                {
                    "body": "tomorrow",
                    "start": 37,
                    "value": {
                        "values": [
                            {
                                "value": "2018-11-14T00:00:00.000-08:00",
                                "grain": "day",
                                "type": "value",
                            }
                        ],
                        "value": "2018-11-14T00:00:00.000-08:00",
                        "grain": "day",
                        "type": "value",
                    },
                    "end": 45,
                    "dim": "time",
                    "latent": False,
                },
            ],
        )

        messages = [
            Message(data={TEXT: "Today is the 5th of May. Let us meet tomorrow."})
        ]
        parsed_messages = duckling.process(messages)

        assert len(parsed_messages) == 1
        entities = parsed_messages[0].get("entities")
        assert len(entities) == 4


def test_duckling_entity_extractor_with_one_extracted_date(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    config = {
        **DucklingEntityExtractorComponent.get_default_config(),
        "dimensions": ["time"],
        "timezone": "UTC",
        "url": "http://localhost:8000",
    }
    duckling = DucklingEntityExtractorComponent.create(
        config, default_model_storage, Resource("duckling"), default_execution_context,
    )

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://localhost:8000/parse",
            json=[
                {
                    "body": "tomorrow",
                    "start": 12,
                    "value": {
                        "values": [
                            {
                                "value": "2013-10-13T00:00:00.000Z",
                                "grain": "day",
                                "type": "value",
                            }
                        ],
                        "value": "2013-10-13T00:00:00.000Z",
                        "grain": "day",
                        "type": "value",
                    },
                    "end": 20,
                    "dim": "time",
                    "latent": False,
                }
            ],
        )

        # 1381536182 == 2013/10/12 02:03:02
        messages = [Message(data={TEXT: "Let us meet tomorrow."}, time=1381536182)]
        duckling.process(messages)

        assert len(messages) == 1
        entities = messages[0].get("entities")
        assert len(entities) == 1
        assert entities[0]["text"] == "tomorrow"
        assert entities[0]["value"] == "2013-10-13T00:00:00.000Z"


def test_duckling_entity_extractor_dimension_filtering(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    config = {
        **DucklingEntityExtractorComponent.get_default_config(),
        "dimensions": ["number"],
        "url": "http://localhost:8000",
    }
    duckling_number = DucklingEntityExtractorComponent.create(
        config, default_model_storage, Resource("duckling"), default_execution_context
    )

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://localhost:8000/parse",
            json=[
                {
                    "body": "Yesterday",
                    "start": 0,
                    "value": {
                        "values": [
                            {
                                "value": "2019-02-28T00:00:00.000+01:00",
                                "grain": "day",
                                "type": "value",
                            }
                        ],
                        "value": "2019-02-28T00:00:00.000+01:00",
                        "grain": "day",
                        "type": "value",
                    },
                    "end": 9,
                    "dim": "time",
                },
                {
                    "body": "5",
                    "start": 21,
                    "value": {"value": 5, "type": "value"},
                    "end": 22,
                    "dim": "number",
                },
            ],
        )

        messages = [Message(data={TEXT: "Yesterday there were 5 people in a room"})]
        duckling_number.process(messages)
        assert len(messages) == 1
        entities = messages[0].get("entities")
        assert len(entities) == 1
        assert entities[0]["text"] == "5"
        assert entities[0]["value"] == 5
