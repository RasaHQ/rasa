import pytest

from rasa_core.interpreter import RegexInterpreter, INTENT_MESSAGE_PREFIX


@pytest.fixture
def regex_interpreter():
    return RegexInterpreter()


def test_regex_interpreter_intent(loop, regex_interpreter):
    text = INTENT_MESSAGE_PREFIX + 'my_intent'
    result = loop.run_until_complete(regex_interpreter.parse(text))
    assert result['text'] == text
    assert len(result['intent_ranking']) == 1
    assert (result['intent']['name'] ==
            result['intent_ranking'][0]['name'] ==
            'my_intent')
    assert (result['intent']['confidence'] ==
            result['intent_ranking'][0]['confidence'] ==
            pytest.approx(1.0))
    assert len(result['entities']) == 0


def test_regex_interpreter_entities(loop, regex_interpreter):
    text = INTENT_MESSAGE_PREFIX + 'my_intent{"foo":"bar"}'
    result = loop.run_until_complete(regex_interpreter.parse(text))
    assert result['text'] == text
    assert len(result['intent_ranking']) == 1
    assert (result['intent']['name'] ==
            result['intent_ranking'][0]['name'] ==
            'my_intent')
    assert (result['intent']['confidence'] ==
            result['intent_ranking'][0]['confidence'] ==
            pytest.approx(1.0))
    assert len(result['entities']) == 1
    assert result["entities"][0]["entity"] == "foo"
    assert result["entities"][0]["value"] == "bar"


def test_regex_interpreter_confidence(loop, regex_interpreter):
    text = INTENT_MESSAGE_PREFIX + 'my_intent@0.5'
    result = loop.run_until_complete(regex_interpreter.parse(text))
    assert result['text'] == text
    assert len(result['intent_ranking']) == 1
    assert (result['intent']['name'] ==
            result['intent_ranking'][0]['name'] ==
            'my_intent')
    assert (result['intent']['confidence'] ==
            result['intent_ranking'][0]['confidence'] ==
            pytest.approx(0.5))
    assert len(result['entities']) == 0


def test_regex_interpreter_confidence_and_entities(loop, regex_interpreter):
    text = INTENT_MESSAGE_PREFIX + 'my_intent@0.5{"foo":"bar"}'
    result = loop.run_until_complete(regex_interpreter.parse(text))
    assert result['text'] == text
    assert len(result['intent_ranking']) == 1
    assert (result['intent']['name'] ==
            result['intent_ranking'][0]['name'] ==
            'my_intent')
    assert (result['intent']['confidence'] ==
            result['intent_ranking'][0]['confidence'] ==
            pytest.approx(0.5))
    assert len(result['entities']) == 1
    assert result["entities"][0]["entity"] == "foo"
    assert result["entities"][0]["value"] == "bar"


def test_regex_interpreter_adds_intent_prefix(loop, regex_interpreter):
    r = loop.run_until_complete(
        regex_interpreter.parse('mood_greet{"name": "rasa"}'))

    assert r.get("text") == '/mood_greet{"name": "rasa"}'
