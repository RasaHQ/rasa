from typing import Any, List
from rasa_sdk import Tracker
import abc
from rasa.shared.nlu.state_machine.state_machine_models import Intent, Slot


class Condition(abc.ABC):
    @abc.abstractmethod
    def is_valid(self, tracker: Tracker):
        raise NotImplementedError()


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
        all([tracker.slots.get(slot.name) for slot in self.slots])


class SlotEqualsCondition(Condition):
    slot: Slot
    value: Any

    def __init__(self, slot: Slot, value: Any):
        self.slot = slot
        self.value = value

    def is_valid(self, tracker: Tracker):
        tracker.slots.get(self.slot.name) == self.value


class AndCondition(Condition):
    conditions: List[Condition]

    def __init__(self, conditions: List[Condition]):
        self.conditions = conditions

    def is_valid(self, tracker: Tracker):
        return all(
            [condition.is_valid(tracker) for condition in self.conditions]
        )


class OrCondition(Condition):
    conditions: List[Condition]

    def __init__(self, conditions: List[Condition]):
        self.conditions = conditions

    def is_valid(self, tracker: Tracker):
        return any(
            [condition.is_valid(tracker) for condition in self.conditions]
        )
