from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from rasa.shared.exceptions import YamlException
from rasa.shared.nlu.constants import INTENT, INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY
from rasa.shared.nlu.training_data.message import Message


@dataclass
class NLUTrigger:
    """Condition for triggering of a flow through intents with confidence thresholds."""

    intent: str
    """The intent to trigger the flow."""
    confidence_threshold: float
    """The minimum threshold of confidence for the intent."""

    def is_triggered(self, intent: str, confidence: float) -> bool:
        """Check if condition is triggered by the given intent and confidence.

        Args:
            intent: The intent to check.
            confidence: The confidence to check.

        Returns:
            Whether the trigger condition is triggered
            by the given intent and confidence.
        """
        return self.intent == intent and confidence >= self.confidence_threshold


@dataclass
class NLUTriggers:
    """List of nlu triggers, e.g. conditions to start a flow through intents."""

    trigger_conditions: List[NLUTrigger]
    """The trigger conditions of the flow."""

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> Optional["NLUTriggers"]:
        """Create an NLUTriggers object from serialized data.

        Args:
            data: data for an NLUTriggers object in a serialized format.

        Returns:
            An NLUTriggers object.

        Raises:
            YamlException: If data format is not as expected.
        """
        if not data:
            return None

        trigger_conditions = []

        for nlu_trigger_config in data:
            if not isinstance(nlu_trigger_config, Dict):
                raise YamlException("NLUTrigger definition not valid. Expected 'dict'.")
            for value in nlu_trigger_config.values():
                if isinstance(value, str):
                    # syntax '- intent: some_intent' was used
                    trigger_conditions.append(NLUTrigger(value, 0.0))
                elif isinstance(value, Dict):
                    # syntax
                    # '- intent:
                    #      name: some_intent'
                    # was used

                    # if no confidence threshold is set, use 0.0 as default
                    confidence = value.get("confidence_threshold", 0.0)

                    trigger_conditions.append(
                        NLUTrigger(value["name"], float(confidence))
                    )

        return NLUTriggers(trigger_conditions=trigger_conditions)

    def as_json(self) -> List[Dict[str, Any]]:
        """Returns the nlu trigger conditions as a list.

        Returns:
            The nlu trigger conditions as a list.
        """
        dump = []

        for trigger_condition in self.trigger_conditions:
            dump.append(
                {
                    "intent": {
                        "name": trigger_condition.intent,
                        "confidence_threshold": trigger_condition.confidence_threshold,
                    }
                }
            )

        return dump

    def is_triggered(self, message: Message) -> bool:
        """Returns whether the flow is triggered by the given intent and threshold.

        Args:
            message: The user message to check.

        Returns:
            Whether the flow step is triggered by the given intent and threshold.
        """
        if not message.get(INTENT) or not message.get(INTENT)[INTENT_NAME_KEY]:
            return False

        intent: str = message.get(INTENT)[INTENT_NAME_KEY]
        confidence: float = message.get(INTENT)[PREDICTED_CONFIDENCE_KEY]

        return any(
            trigger_condition.is_triggered(intent, confidence)
            for trigger_condition in self.trigger_conditions
        )
