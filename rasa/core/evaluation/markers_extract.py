from typing import Dict, Text, List
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event


def extract_markers(tracker: DialogueStateTracker, marker_config: Dict):
    # TODO call the code below and test this
    applied_events = tracker.applied_events()
    for marker_dict in marker_config:
        marker = Marker(
            name=marker_dict.get("marker"),
            operator=marker_dict.get("operator"),
            condition=marker_dict.get("condition"),
        )
        Marker.does_marker_apply(applied_events)
        # TODO save the marker to a file 


class Marker:
    def __init__(self, name: Text, operator: Text, condition: List[Dict[Text, List]]):
        self.name = name
        self.operator = operator
        self.condition = condition
        # list of conditions
        # need to decide if AND or operator applies

        if self.operator in ["AND", "OR"]:
            # group similar atomic conditions together since sequence doesn't matter
            # for AND and OR
            self.slot_set = Marker._get_values(self.condition, "slot_set")
            self.slot_not_set = Marker._get_values(self.condition, "slot_not_set")
            self.action_executed = Marker._get_values(self.condition, "action_executed")
            self.action_not_executed = Marker._get_values(
                self.condition, "action_not_executed"
            )
            self.intent_detected = Marker._get_values(self.condition, "intent_detected")
            self.intent_not_detected = Marker._get_values(
                self.condition, "intent_not_detected"
            )

        self.preceding_user_turns = []
        self.timestamps = []

    @staticmethod
    def _get_values(
        condition: List[Dict[Text, List]], atomic_condition: Text
    ) -> List[Text]:
        """Gets all the values listed under the same atomic condition label."""
        all_items = []
        for item in condition:
            if atomic_condition in item.keys():
                all_items.extend(item.get(atomic_condition))
        return list(set(all_items))

    def does_marker_apply(self, events) -> bool:

        if self.operator == "AND":
            # TODO check that all apply
            return True

        elif self.operator == "OR":
            # TODO check that any apply
            return True

        elif self.operator == "SEQ":
            # TODO check that all apply in the same order
            return True

        else:
            raise RasaException(
                "Invalid marker operator - '{self.operator}' was given, \
                options 'AND', 'OR', and 'SEQ' exist."
            )

    def get_relevant_events(self, events) -> List[Event]:
        """Get an ordered list of relevant events"""
        relevant_events = []

        for e in events:
            if isinstance(e, SlotSet):
                if e.key in self.slot_set or e.key in self.slot_not_set:
                    relevant_events.append(e)

            if isinstance(e, ActionExecuted):
                if (
                    e.action_name in self.action_executed
                    or e.action_name in self.action_not_executed
                ):
                    relevant_events.append(e)

            if isinstance(e, UserUttered):
                if (
                    e.intent in self.intent_detected
                    or e.intent in self.intent_not_detected
                ):
                    relevant_events.append(e)

        return relevant_events

    def check_and(self, events):
        """Checks that the AND condition applies"""
        return

    def check_or(self, events):
        """Checks that the OR condition applies."""
        user_turn_index = 0
        slot_set = []
        action_executed = []
        intent_detected = []
        satisfied = False

        # check the positive atomic conditions (slot_set, action_executed, etc.)
        for e in events:
            if isinstance(e, SlotSet):
                if e.key in self.slot_set and e.value:
                    self._mark_event(e.timestamp, user_turn_index)
                    satisfied = True
                if e.key in self.slot_not_set and e.value:
                    slot_set.append(e.key)

            if isinstance(e, ActionExecuted):
                if e.action_name in self.action_executed:
                    self._mark_event(e.timestamp, user_turn_index)
                    satisfied = True
                if e.action_name in self.action_not_executed:
                    action_executed.append(e.action_name)

            if isinstance(e, UserUttered):
                user_turn_index += 1
                if e.intent.get("name") in self.intent_detected:
                    self._mark_event(e.timestamp, user_turn_index)
                    satisfied = True
                if e.intent.get("name") in self.intent_not_detected:
                    intent_detected.append(e.intent.get("name"))

        # check the negative atomic conditions (slot_not_set, action_not_executed, etc.)
        e = events[-1]
        for slot in self.slot_not_set:
            if slot not in slot_set:
                self._mark_event(e.timestamp, user_turn_index)
                satisfied = True

        for action in self.action_not_executed:
            if action not in action_executed:
                self._mark_event(e.timestamp, user_turn_index)
                satisfied = True

        for intent in self.intent_not_detected:
            if intent not in intent_detected:
                self._mark_event(e.timestamp, user_turn_index)
                satisfied = True

        return satisfied

    def check_seq(self, events):
        """Checks that the SEQ condition applies."""
        return

    def _mark_event(self, timestamp, user_turn_index):
        self.timestamps.append(timestamp)
        self.preceding_user_turns.append(user_turn_index)
