from __future__ import annotations

from dataclasses import dataclass
from typing import Text, List, runtime_checkable, Protocol, Dict, Any

from rasa.shared.core.flows.flow_step import FlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT_NAME_KEY, ENTITY_ATTRIBUTE_TYPE


@dataclass
class TriggerCondition:
    """Represents the configuration of a trigger condition."""

    intent: Text
    """The intent to trigger the flow."""
    entities: List[Text]
    """The entities to trigger the flow."""

    def is_triggered(self, intent: Text, entities: List[Text]) -> bool:
        """Check if condition is triggered by the given intent and entities.

        Args:
            intent: The intent to check.
            entities: The entities to check.

        Returns:
            Whether the trigger condition is triggered by the given intent and entities.
        """
        if self.intent != intent:
            return False
        if len(self.entities) == 0:
            return True
        return all(entity in entities for entity in self.entities)


@runtime_checkable
class StepThatCanStartAFlow(Protocol):
    """Represents a step that can start a flow."""

    def is_triggered(self, tracker: DialogueStateTracker) -> bool:
        """Check if a flow should be started for the tracker

        Args:
            tracker: The tracker to check.

        Returns:
            Whether a flow should be started for the tracker.
        """
        ...


@dataclass
class UserMessageStep(FlowStep, StepThatCanStartAFlow):
    """Represents the configuration of an intent flow step."""

    trigger_conditions: List[TriggerCondition]
    """The trigger conditions of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> UserMessageStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)

        trigger_conditions = []
        if "intent" in flow_step_config:
            trigger_conditions.append(
                TriggerCondition(
                    intent=flow_step_config["intent"],
                    entities=flow_step_config.get("entities", []),
                )
            )
        elif "or" in flow_step_config:
            for trigger_condition in flow_step_config["or"]:
                trigger_conditions.append(
                    TriggerCondition(
                        intent=trigger_condition.get("intent", ""),
                        entities=trigger_condition.get("entities", []),
                    )
                )

        return UserMessageStep(
            trigger_conditions=trigger_conditions,
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()

        if len(self.trigger_conditions) == 1:
            dump["intent"] = self.trigger_conditions[0].intent
            if self.trigger_conditions[0].entities:
                dump["entities"] = self.trigger_conditions[0].entities
        elif len(self.trigger_conditions) > 1:
            dump["or"] = [
                {
                    "intent": trigger_condition.intent,
                    "entities": trigger_condition.entities,
                }
                for trigger_condition in self.trigger_conditions
            ]

        return dump

    def is_triggered(self, tracker: DialogueStateTracker) -> bool:
        """Returns whether the flow step is triggered by the given intent and entities.

        Args:
            intent: The intent to check.
            entities: The entities to check.

        Returns:
            Whether the flow step is triggered by the given intent and entities.
        """
        if not tracker.latest_message:
            return False

        intent: Text = tracker.latest_message.intent.get(INTENT_NAME_KEY, "")
        entities: List[Text] = [
            e.get(ENTITY_ATTRIBUTE_TYPE, "") for e in tracker.latest_message.entities
        ]
        return any(
            trigger_condition.is_triggered(intent, entities)
            for trigger_condition in self.trigger_conditions
        )

    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "intent"
