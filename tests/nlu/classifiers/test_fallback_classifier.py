import copy
from typing import Dict

import pytest

from rasa.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core.constants import DEFAULT_NLU_FALLBACK_THRESHOLD
from rasa.nlu.classifiers.fallback_classifier import (
    FallbackClassifier,
    THRESHOLD_KEY,
    AMBIGUITY_THRESHOLD_KEY,
)
from rasa.nlu.training_data import Message
from rasa.nlu.constants import (
    INTENT_RANKING_KEY,
    INTENT,
    PREDICTED_CONFIDENCE_KEY,
    INTENT_NAME_KEY,
)


@pytest.mark.parametrize(
    "message, component_config",
    [
        (
            Message(
                "some message",
                data={
                    INTENT: {
                        INTENT_NAME_KEY: "greet",
                        PREDICTED_CONFIDENCE_KEY: 0.234891876578331,
                    },
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.234891876578331,
                        },
                        {
                            INTENT_NAME_KEY: "stop",
                            PREDICTED_CONFIDENCE_KEY: 0.5 - 0.0001,
                        },
                        {INTENT_NAME_KEY: "affirm", PREDICTED_CONFIDENCE_KEY: 0},
                        {INTENT_NAME_KEY: "inform", PREDICTED_CONFIDENCE_KEY: -100},
                        {
                            INTENT_NAME_KEY: "deny",
                            PREDICTED_CONFIDENCE_KEY: 0.0879683718085289,
                        },
                    ],
                },
            ),
            {THRESHOLD_KEY: 0.5},
        ),
        (
            Message(
                "some message",
                data={
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                    INTENT_RANKING_KEY: [
                        {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.9},
                    ],
                },
            ),
            {THRESHOLD_KEY: 0.5, AMBIGUITY_THRESHOLD_KEY: 0.1},
        ),
        (
            Message(
                "some message",
                data={
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                    INTENT_RANKING_KEY: [
                        {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.5},
                    ],
                },
            ),
            {THRESHOLD_KEY: 0.5, AMBIGUITY_THRESHOLD_KEY: 0.51},
        ),
    ],
)
def test_predict_fallback_intent(message: Message, component_config: Dict):
    old_message_state = copy.deepcopy(message)
    classifier = FallbackClassifier(component_config=component_config)
    classifier.process(message)

    expected_intent = {
        INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME,
        PREDICTED_CONFIDENCE_KEY: 1.0,
    }
    assert message.data[INTENT] == expected_intent

    old_intent_ranking = old_message_state.data[INTENT_RANKING_KEY]
    current_intent_ranking = message.data[INTENT_RANKING_KEY]

    assert len(current_intent_ranking) == len(old_intent_ranking) + 1
    assert all(item in current_intent_ranking for item in old_intent_ranking)
    assert current_intent_ranking[0] == expected_intent


@pytest.mark.parametrize(
    "message, component_config",
    [
        (
            Message(
                "some message",
                data={
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 0.5},
                    INTENT_RANKING_KEY: [
                        {
                            INTENT_NAME_KEY: "greet",
                            PREDICTED_CONFIDENCE_KEY: 0.234891876578331,
                        },
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.1},
                        {INTENT_NAME_KEY: "affirm", PREDICTED_CONFIDENCE_KEY: 0},
                        {INTENT_NAME_KEY: "inform", PREDICTED_CONFIDENCE_KEY: -100},
                        {
                            INTENT_NAME_KEY: "deny",
                            PREDICTED_CONFIDENCE_KEY: 0.0879683718085289,
                        },
                    ],
                },
            ),
            {THRESHOLD_KEY: 0.5},
        ),
        (
            Message(
                "some message",
                data={
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                    INTENT_RANKING_KEY: [
                        {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.89},
                    ],
                },
            ),
            {THRESHOLD_KEY: 0.5, AMBIGUITY_THRESHOLD_KEY: 0.1},
        ),
    ],
)
def test_not_predict_fallback_intent(message: Message, component_config: Dict):
    old_message_state = copy.deepcopy(message)

    classifier = FallbackClassifier(component_config=component_config)
    classifier.process(message)

    assert message == old_message_state


def test_defaults():
    classifier = FallbackClassifier({})

    assert classifier.component_config[THRESHOLD_KEY] == DEFAULT_NLU_FALLBACK_THRESHOLD
    assert classifier.component_config[AMBIGUITY_THRESHOLD_KEY] == 0.1
