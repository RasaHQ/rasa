import pytest
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.shared.nlu.interpreter import RegexInterpreter


async def test_regex_interpreter_intent():
    text = INTENT_MESSAGE_PREFIX + "my_intent"
    result = await RegexInterpreter().parse(text)
    assert result["text"] == text
    assert len(result["intent_ranking"]) == 1
    assert (
        result["intent"][INTENT_NAME_KEY]
        == result["intent_ranking"][0][INTENT_NAME_KEY]
        == "my_intent"
    )
    assert (
        result["intent"]["confidence"]
        == result["intent_ranking"][0]["confidence"]
        == pytest.approx(1.0)
    )
    assert len(result["entities"]) == 0


async def test_regex_interpreter_entities():
    text = INTENT_MESSAGE_PREFIX + 'my_intent{"foo":"bar"}'
    result = await RegexInterpreter().parse(text)
    assert result["text"] == text
    assert len(result["intent_ranking"]) == 1
    assert (
        result["intent"][INTENT_NAME_KEY]
        == result["intent_ranking"][0][INTENT_NAME_KEY]
        == "my_intent"
    )
    assert (
        result["intent"]["confidence"]
        == result["intent_ranking"][0]["confidence"]
        == pytest.approx(1.0)
    )
    assert len(result["entities"]) == 1
    assert result["entities"][0]["entity"] == "foo"
    assert result["entities"][0]["value"] == "bar"


async def test_regex_interpreter_confidence():
    text = INTENT_MESSAGE_PREFIX + "my_intent@0.5"
    result = await RegexInterpreter().parse(text)
    assert result["text"] == text
    assert len(result["intent_ranking"]) == 1
    assert (
        result["intent"][INTENT_NAME_KEY]
        == result["intent_ranking"][0][INTENT_NAME_KEY]
        == "my_intent"
    )
    assert (
        result["intent"]["confidence"]
        == result["intent_ranking"][0]["confidence"]
        == pytest.approx(0.5)
    )
    assert len(result["entities"]) == 0


async def test_regex_interpreter_confidence_and_entities():
    text = INTENT_MESSAGE_PREFIX + 'my_intent@0.5{"foo":"bar"}'
    result = await RegexInterpreter().parse(text)
    assert result["text"] == text
    assert len(result["intent_ranking"]) == 1
    assert (
        result["intent"][INTENT_NAME_KEY]
        == result["intent_ranking"][0][INTENT_NAME_KEY]
        == "my_intent"
    )
    assert (
        result["intent"]["confidence"]
        == result["intent_ranking"][0]["confidence"]
        == pytest.approx(0.5)
    )
    assert len(result["entities"]) == 1
    assert result["entities"][0]["entity"] == "foo"
    assert result["entities"][0]["value"] == "bar"


async def test_regex_interpreter_adds_intent_prefix():
    r = await RegexInterpreter().parse('mood_greet{"name": "rasa"}')

    assert r.get("text") == '/mood_greet{"name": "rasa"}'
