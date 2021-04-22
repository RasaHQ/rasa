import copy
from typing import Dict, Text, Any

import pytest

from rasa.nlu.classifiers import fallback_classifier
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core.constants import DEFAULT_NLU_FALLBACK_THRESHOLD
from rasa.nlu.classifiers.fallback_classifier import (
    FallbackClassifier,
    THRESHOLD_KEY,
    AMBIGUITY_THRESHOLD_KEY,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
)


@pytest.mark.parametrize(
    "message, component_config",
    [
        (
            Message(
                data={
                    TEXT: "some message",
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
                }
            ),
            {THRESHOLD_KEY: 0.5},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                    INTENT_RANKING_KEY: [
                        {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.9},
                    ],
                }
            ),
            {THRESHOLD_KEY: 0.5, AMBIGUITY_THRESHOLD_KEY: 0.1},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                    INTENT_RANKING_KEY: [
                        {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.5},
                    ],
                }
            ),
            {THRESHOLD_KEY: 0.5, AMBIGUITY_THRESHOLD_KEY: 0.51},
        ),
    ],
)
def test_predict_fallback_intent(message: Message, component_config: Dict):
    old_message_state = copy.deepcopy(message)
    classifier = FallbackClassifier(component_config=component_config)
    classifier.process(message)

    confidence = 1 - old_message_state.data[INTENT][PREDICTED_CONFIDENCE_KEY]
    expected_intent = {
        INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME,
        PREDICTED_CONFIDENCE_KEY: confidence,
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
                data={
                    TEXT: "some message",
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
                }
            ),
            {THRESHOLD_KEY: 0.5},
        ),
        (
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                    INTENT_RANKING_KEY: [
                        {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1},
                        {INTENT_NAME_KEY: "stop", PREDICTED_CONFIDENCE_KEY: 0.89},
                    ],
                }
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


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ({INTENT: {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME}}, True),
        ({INTENT: {INTENT_NAME_KEY: "some other intent"}}, False),
    ],
)
def test_is_fallback_classifier_prediction(prediction: Dict[Text, Any], expected: bool):
    assert fallback_classifier.is_fallback_classifier_prediction(prediction) == expected


@pytest.mark.parametrize(
    "prediction, expected",
    [
        (
            {INTENT: {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME}},
            {INTENT: {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME}},
        ),
        (
            {
                INTENT: {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME},
                INTENT_RANKING_KEY: [],
            },
            {
                INTENT: {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME},
                INTENT_RANKING_KEY: [],
            },
        ),
        (
            {
                INTENT: {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME},
                INTENT_RANKING_KEY: [
                    {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME}
                ],
            },
            {
                INTENT: {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME},
                INTENT_RANKING_KEY: [
                    {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME}
                ],
            },
        ),
        (
            {
                INTENT: {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME},
                INTENT_RANKING_KEY: [
                    {INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME},
                    {INTENT_NAME_KEY: "other", PREDICTED_CONFIDENCE_KEY: 123},
                    {INTENT_NAME_KEY: "other2", PREDICTED_CONFIDENCE_KEY: 12},
                ],
            },
            {
                INTENT: {INTENT_NAME_KEY: "other", PREDICTED_CONFIDENCE_KEY: 123},
                INTENT_RANKING_KEY: [
                    {INTENT_NAME_KEY: "other", PREDICTED_CONFIDENCE_KEY: 123},
                    {INTENT_NAME_KEY: "other2", PREDICTED_CONFIDENCE_KEY: 12},
                ],
            },
        ),
    ],
)
def test_undo_fallback_prediction(
    prediction: Dict[Text, Any], expected: Dict[Text, Any]
):
    assert fallback_classifier.undo_fallback_prediction(prediction) == expected
