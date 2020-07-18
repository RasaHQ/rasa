import copy

from rasa.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core.constants import DEFAULT_NLU_FALLBACK_THRESHOLD
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier, THRESHOLD_KEY
from rasa.nlu.training_data import Message
from rasa.nlu.constants import INTENT_RANKING_KEY, INTENT, INTENT_CONFIDENCE_KEY


def test_predict_fallback_intent():
    threshold = 0.5
    message = Message(
        "some message",
        data={
            INTENT: {"name": "greet", INTENT_CONFIDENCE_KEY: 0.234891876578331},
            INTENT_RANKING_KEY: [
                {"name": "greet", INTENT_CONFIDENCE_KEY: 0.234891876578331},
                {"name": "stop", INTENT_CONFIDENCE_KEY: threshold - 0.0001},
                {"name": "affirm", INTENT_CONFIDENCE_KEY: 0},
                {"name": "inform", INTENT_CONFIDENCE_KEY: -100},
                {"name": "deny", INTENT_CONFIDENCE_KEY: 0.0879683718085289},
            ],
        },
    )
    old_message_state = copy.deepcopy(message)

    classifier = FallbackClassifier(component_config={THRESHOLD_KEY: threshold})
    classifier.process(message)

    expected_intent = {
        "name": DEFAULT_NLU_FALLBACK_INTENT_NAME,
        INTENT_CONFIDENCE_KEY: 1.0,
    }
    assert message.data[INTENT] == expected_intent

    old_intent_ranking = old_message_state.data[INTENT_RANKING_KEY]
    current_intent_ranking = message.data[INTENT_RANKING_KEY]

    assert len(current_intent_ranking) == len(old_intent_ranking) + 1
    assert all(item in current_intent_ranking for item in old_intent_ranking)
    assert current_intent_ranking[0] == expected_intent


def test_not_predict_fallback_intent():
    threshold = 0.5
    message = Message(
        "some message",
        data={
            INTENT: {"name": "greet", INTENT_CONFIDENCE_KEY: threshold},
            INTENT_RANKING_KEY: [
                {"name": "greet", INTENT_CONFIDENCE_KEY: 0.234891876578331},
                {"name": "stop", INTENT_CONFIDENCE_KEY: 0.1},
                {"name": "affirm", INTENT_CONFIDENCE_KEY: 0},
                {"name": "inform", INTENT_CONFIDENCE_KEY: -100},
                {"name": "deny", INTENT_CONFIDENCE_KEY: 0.0879683718085289},
            ],
        },
    )
    old_message_state = copy.deepcopy(message)

    classifier = FallbackClassifier(component_config={THRESHOLD_KEY: threshold})
    classifier.process(message)

    assert message == old_message_state


def test_default_threshold():
    classifier = FallbackClassifier({})

    assert classifier.component_config[THRESHOLD_KEY] == DEFAULT_NLU_FALLBACK_THRESHOLD
