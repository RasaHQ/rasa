import responses

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message


def test_duckling_entity_extractor(component_builder):
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

        _config = RasaNLUModelConfig({"pipeline": [{"name": "DucklingHTTPExtractor"}]})
        _config.set_component_attr(
            0, dimensions=["time"], timezone="UTC", url="http://localhost:8000"
        )
        duckling = component_builder.create_component(_config.for_component(0), _config)
        message = Message("Today is the 5th of May. Let us meet tomorrow.")
        duckling.process(message)
        entities = message.get("entities")
        assert len(entities) == 4

    # Test duckling with a defined date

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
        message = Message("Let us meet tomorrow.", time="1381536182")
        duckling.process(message)
        entities = message.get("entities")
        assert len(entities) == 1
        assert entities[0]["text"] == "tomorrow"
        assert entities[0]["value"] == "2013-10-13T00:00:00.000Z"

        # Test dimension filtering includes only specified dimensions
        _config = RasaNLUModelConfig({"pipeline": [{"name": "DucklingHTTPExtractor"}]})
        _config.set_component_attr(
            0, dimensions=["number"], url="http://localhost:8000"
        )
        duckling_number = component_builder.create_component(
            _config.for_component(0), _config
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

        message = Message("Yesterday there were 5 people in a room")
        duckling_number.process(message)
        entities = message.get("entities")

        assert len(entities) == 1
        assert entities[0]["text"] == "5"
        assert entities[0]["value"] == 5


def test_duckling_entity_extractor_and_synonyms(component_builder):
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "DucklingHTTPExtractor"},
                {"name": "EntitySynonymMapper"},
            ]
        }
    )
    _config.set_component_attr(0, dimensions=["number"])
    duckling = component_builder.create_component(_config.for_component(0), _config)
    synonyms = component_builder.create_component(_config.for_component(1), _config)
    message = Message("He was 6 feet away")
    duckling.process(message)
    # checks that the synonym processor
    # can handle entities that have int values
    synonyms.process(message)
    assert message is not None
