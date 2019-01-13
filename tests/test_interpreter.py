import pytest

import rasa_core
from rasa_core.interpreter import RegexInterpreter, INTENT_MESSAGE_PREFIX


def test_regex_interpreter():
    interp = RegexInterpreter()

    text = INTENT_MESSAGE_PREFIX + 'my_intent'
    result = interp.parse(text)
    assert result['text'] == text
    assert len(result['intent_ranking']) == 1
    assert result['intent']['name'] == \
        result['intent_ranking'][0]['name'] == \
        'my_intent'
    assert result['intent']['confidence'] == \
        result['intent_ranking'][0]['confidence'] == \
        pytest.approx(1.0)
    assert len(result['entities']) == 0

    text = INTENT_MESSAGE_PREFIX + 'my_intent{"foo":"bar"}'
    result = interp.parse(text)
    assert result['text'] == text
    assert len(result['intent_ranking']) == 1
    assert result['intent']['name'] == \
        result['intent_ranking'][0]['name'] == \
        'my_intent'
    assert result['intent']['confidence'] == \
        result['intent_ranking'][0]['confidence'] == \
        pytest.approx(1.0)
    assert len(result['entities']) == 1
    assert result["entities"][0]["entity"] == "foo"
    assert result["entities"][0]["value"] == "bar"

    text = INTENT_MESSAGE_PREFIX + 'my_intent@0.5'
    result = interp.parse(text)
    assert result['text'] == text
    assert len(result['intent_ranking']) == 1
    assert result['intent']['name'] == \
        result['intent_ranking'][0]['name'] == \
        'my_intent'
    assert result['intent']['confidence'] == \
        result['intent_ranking'][0]['confidence'] == \
        pytest.approx(0.5)
    assert len(result['entities']) == 0

    text = INTENT_MESSAGE_PREFIX + 'my_intent@0.5{"foo":"bar"}'
    result = interp.parse(text)
    assert result['text'] == text
    assert len(result['intent_ranking']) == 1
    assert result['intent']['name'] == \
        result['intent_ranking'][0]['name'] == \
        'my_intent'
    assert result['intent']['confidence'] == \
        result['intent_ranking'][0]['confidence'] == \
        pytest.approx(0.5)
    assert len(result['entities']) == 1
    assert result["entities"][0]["entity"] == "foo"
    assert result["entities"][0]["value"] == "bar"


def test_regex_interpreter_adds_intent_prefix():
    interpreter = RegexInterpreter()
    r = interpreter.parse('mood_greet{"name": "rasa"}')

    assert r.get("text") == '/mood_greet{"name": "rasa"}'
