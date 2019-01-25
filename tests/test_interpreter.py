import pytest

import rasa_core
from rasa_core.interpreter import (RegexInterpreter,
                                   INTENT_MESSAGE_PREFIX,
                                   RasaNLUHttpInterpreter)
from rasa_core.utils import EndpointConfig
from httpretty import httpretty


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


def test_http_interpreter():

    httpretty.register_uri(httpretty.GET,
                           'https://interpreter.com/parse')

    endpoint = EndpointConfig('https://interpreter.com')
    httpretty.enable()
    interpreter = RasaNLUHttpInterpreter(endpoint=endpoint)
    interpreter.parse(text='message_text', message_id='1134')

    query = httpretty.last_request.querystring
    httpretty.disable()
    response = {'project': ['default'],
                'q': ['message_text'],
                'message_id': ['1134']}

    assert query == response
