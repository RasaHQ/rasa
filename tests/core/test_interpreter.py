import pytest
from aioresponses import aioresponses

from rasa.core.interpreter import (
    INTENT_MESSAGE_PREFIX,
    RasaNLUHttpInterpreter,
    RegexInterpreter,
)
from rasa.utils.endpoints import EndpointConfig
from tests.utilities import latest_request, json_of_latest_request


@pytest.fixture
def regex_interpreter():
    return RegexInterpreter()


async def test_regex_interpreter_intent(regex_interpreter):
    text = INTENT_MESSAGE_PREFIX + "my_intent"
    result = await regex_interpreter.parse(text)
    assert result["text"] == text
    assert len(result["intent_ranking"]) == 1
    assert (
        result["intent"]["name"] == result["intent_ranking"][0]["name"] == "my_intent"
    )
    assert (
        result["intent"]["confidence"]
        == result["intent_ranking"][0]["confidence"]
        == pytest.approx(1.0)
    )
    assert len(result["entities"]) == 0


async def test_regex_interpreter_entities(regex_interpreter):
    text = INTENT_MESSAGE_PREFIX + 'my_intent{"foo":"bar"}'
    result = await regex_interpreter.parse(text)
    assert result["text"] == text
    assert len(result["intent_ranking"]) == 1
    assert (
        result["intent"]["name"] == result["intent_ranking"][0]["name"] == "my_intent"
    )
    assert (
        result["intent"]["confidence"]
        == result["intent_ranking"][0]["confidence"]
        == pytest.approx(1.0)
    )
    assert len(result["entities"]) == 1
    assert result["entities"][0]["entity"] == "foo"
    assert result["entities"][0]["value"] == "bar"


async def test_regex_interpreter_confidence(regex_interpreter):
    text = INTENT_MESSAGE_PREFIX + "my_intent@0.5"
    result = await regex_interpreter.parse(text)
    assert result["text"] == text
    assert len(result["intent_ranking"]) == 1
    assert (
        result["intent"]["name"] == result["intent_ranking"][0]["name"] == "my_intent"
    )
    assert (
        result["intent"]["confidence"]
        == result["intent_ranking"][0]["confidence"]
        == pytest.approx(0.5)
    )
    assert len(result["entities"]) == 0


async def test_regex_interpreter_confidence_and_entities(regex_interpreter):
    text = INTENT_MESSAGE_PREFIX + 'my_intent@0.5{"foo":"bar"}'
    result = await regex_interpreter.parse(text)
    assert result["text"] == text
    assert len(result["intent_ranking"]) == 1
    assert (
        result["intent"]["name"] == result["intent_ranking"][0]["name"] == "my_intent"
    )
    assert (
        result["intent"]["confidence"]
        == result["intent_ranking"][0]["confidence"]
        == pytest.approx(0.5)
    )
    assert len(result["entities"]) == 1
    assert result["entities"][0]["entity"] == "foo"
    assert result["entities"][0]["value"] == "bar"


async def test_regex_interpreter_adds_intent_prefix(regex_interpreter):
    r = await regex_interpreter.parse('mood_greet{"name": "rasa"}')

    assert r.get("text") == '/mood_greet{"name": "rasa"}'


async def test_http_interpreter():
    with aioresponses() as mocked:
        mocked.post("https://example.com/model/parse")

        endpoint = EndpointConfig("https://example.com")
        interpreter = RasaNLUHttpInterpreter(endpoint=endpoint)
        await interpreter.parse(text="message_text", message_id="message_id")

        r = latest_request(mocked, "POST", "https://example.com/model/parse")

        query = json_of_latest_request(r)
        response = {"text": "message_text", "token": None, "message_id": "message_id"}

        assert query == response
