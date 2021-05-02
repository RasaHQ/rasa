from typing import Any, List
from rasa_sdk import Tracker
import abc
from rasa.shared.nlu.state_machine.state_machine_models import Intent, Slot
from rasa.shared.nlu.state_machine.condition import Condition


class OnEntryCondition(Condition):
    def is_valid(self, tracker: Tracker):
        return tracker.active_loop_name == None


class IntentCondition(Condition):
    intent: Intent

    def __init__(self, intent: Intent):
        self.intent = intent

    def is_valid(self, tracker: Tracker):
        last_intent_name = tracker.latest_message.intent.get("name")
        return self.intent.name == last_intent_name


class SlotsFilledCondition(Condition):
    slots: List[Slot]

    def __init__(self, slots: [Slot]):
        self.slots = slots

    def is_valid(self, tracker: Tracker):
        return all([tracker.slots.get(slot.name).value for slot in self.slots])


class SlotEqualsCondition(Condition):
    slot: Slot
    value: Any

    def __init__(self, slot: Slot, value: Any):
        self.slot = slot
        self.value = value

    def is_valid(self, tracker: Tracker):
        tracker_slot = tracker.slots.get(self.slot.name)

        if tracker_slot:
            return tracker_slot.value == self.value
        else:
            raise RuntimeError("Required slot not found in tracker")


class ConditionWithConditions(abc.ABC):
    @abc.abstractproperty
    def conditions() -> List[Condition]:
        pass

    @property
    def intents() -> List[Intent]:
        all_intents: List[Intent] = []
        for condition in conditions:
            if isinstance(condition, OrCondition):
                all_intents += condition.intents
            elif isinstance(condition, AndCondition):
                all_intents += condition.intents
            elif isinstance(condition, IntentCondition):
                all_intents += condition.intent

        return all_intents


class AndCondition(Condition, ConditionWithConditions):
    @property
    def conditions() -> List[Condition]:
        return self._conditions

    def __init__(self, conditions: List[Condition]):
        self._conditions = conditions

    def is_valid(self, tracker: Tracker):
        return all(
            [condition.is_valid(tracker) for condition in self.conditions]
        )


class OrCondition(Condition, ConditionWithConditions):
    @property
    def conditions() -> List[Condition]:
        return self._conditions

    def __init__(self, conditions: List[Condition]):
        self._conditions = conditions

    def is_valid(self, tracker: Tracker):
        return any(
            [condition.is_valid(tracker) for condition in self.conditions]
        )
