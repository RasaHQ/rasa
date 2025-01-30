from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Text, Tuple, Type, Union

from rasa.core.constants import (
    DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
    DEFAULT_NLU_FALLBACK_THRESHOLD,
)
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
from rasa.shared.nlu.training_data.message import Message

THRESHOLD_KEY = "threshold"
AMBIGUITY_THRESHOLD_KEY = "ambiguity_threshold"

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)
class FallbackClassifier(GraphComponent, IntentClassifier):
    """Handles incoming messages with low NLU confidence."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [IntentClassifier]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            # If all intent confidence scores are beyond this threshold, set the current
            # intent to `FALLBACK_INTENT_NAME`
            THRESHOLD_KEY: DEFAULT_NLU_FALLBACK_THRESHOLD,
            # If the confidence scores for the top two intent predictions are closer
            # than `AMBIGUITY_THRESHOLD_KEY`,
            # then `FALLBACK_INTENT_NAME` is predicted.
            AMBIGUITY_THRESHOLD_KEY: DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Constructs a new fallback classifier."""
        self.component_config = config

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> FallbackClassifier:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:
        """Process a list of incoming messages.

        This is the component's chance to process incoming
        messages. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.

        Args:
            messages: List containing :class:
            `rasa.shared.nlu.training_data.message.Message` to process.
        """
        for message in messages:
            if not self._should_fallback(message):
                continue

            # we assume that the fallback confidence
            # is the same as the fallback threshold
            confidence = self.component_config[THRESHOLD_KEY]
            message.data[INTENT] = _fallback_intent(confidence)
            message.data.setdefault(INTENT_RANKING_KEY, [])
            message.data[INTENT_RANKING_KEY].insert(0, _fallback_intent(confidence))

        return messages

    def _should_fallback(self, message: Message) -> bool:
        """Check if the fallback intent should be predicted.

        Args:
            message: The current message and its intent predictions.

        Returns:
            `True` if the fallback intent should be predicted.
        """
        intent_name = message.data[INTENT].get(INTENT_NAME_KEY)
        below_threshold, nlu_confidence = self._nlu_confidence_below_threshold(message)

        if below_threshold:
            logger.debug(
                f"NLU confidence {nlu_confidence} for intent '{intent_name}' is lower "
                f"than NLU threshold {self.component_config[THRESHOLD_KEY]:.2f}."
            )
            return True

        ambiguous_prediction, confidence_delta = self._nlu_prediction_ambiguous(message)
        if ambiguous_prediction:
            logger.debug(
                f"The difference in NLU confidences "
                f"for the top two intents ({confidence_delta}) is lower than "
                f"the ambiguity threshold "
                f"{self.component_config[AMBIGUITY_THRESHOLD_KEY]:.2f}. Predicting "
                f"intent '{DEFAULT_NLU_FALLBACK_INTENT_NAME}' instead of "
                f"'{intent_name}'."
            )
            return True

        return False

    def _nlu_confidence_below_threshold(self, message: Message) -> Tuple[bool, float]:
        nlu_confidence = message.data[INTENT].get(PREDICTED_CONFIDENCE_KEY)
        return nlu_confidence < self.component_config[THRESHOLD_KEY], nlu_confidence

    def _nlu_prediction_ambiguous(
        self, message: Message
    ) -> Tuple[bool, Optional[float]]:
        intents = message.data.get(INTENT_RANKING_KEY, [])
        if len(intents) >= 2:
            first_confidence = intents[0].get(PREDICTED_CONFIDENCE_KEY, 1.0)
            second_confidence = intents[1].get(PREDICTED_CONFIDENCE_KEY, 1.0)
            difference = first_confidence - second_confidence
            return (
                difference < self.component_config[AMBIGUITY_THRESHOLD_KEY],
                difference,
            )
        return False, None


def _fallback_intent(confidence: float) -> Dict[Text, Union[Text, float]]:
    return {
        INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME,
        PREDICTED_CONFIDENCE_KEY: confidence,
    }


def is_fallback_classifier_prediction(prediction: Dict[Text, Any]) -> bool:
    """Checks if the intent was predicted by the `FallbackClassifier`.

    Args:
        prediction: The prediction of the NLU model.

    Returns:
        `True` if the top classified intent was the fallback intent.
    """
    return (
        prediction.get(INTENT, {}).get(INTENT_NAME_KEY)
        == DEFAULT_NLU_FALLBACK_INTENT_NAME
    )


def undo_fallback_prediction(prediction: Dict[Text, Any]) -> Dict[Text, Any]:
    """Undo the prediction of the fallback intent.

    Args:
        prediction: The prediction of the NLU model.

    Returns:
        The prediction as if the `FallbackClassifier` wasn't present in the pipeline.
        If the fallback intent is the only intent, return the prediction as it was
        provided.
    """
    intent_ranking = prediction.get(INTENT_RANKING_KEY, [])
    if len(intent_ranking) < 2:
        return prediction

    prediction[INTENT] = intent_ranking[1]
    prediction[INTENT_RANKING_KEY] = prediction[INTENT_RANKING_KEY][1:]
    return prediction
