from typing import Dict, Text, List
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event


def extract_markers(tracker: DialogueStateTracker, marker_config: Dict) -> None:
    """Extracts markers."""
    # TODO test this
    applied_events = get_relevant_events(tracker.applied_events())
    for marker_dict in marker_config:
        marker = Marker(
            name=marker_dict.get("marker"),
            operator=marker_dict.get("operator"),
            condition=marker_dict.get("condition"),
        )
        marker.does_marker_apply(applied_events)
        # TODO output the marker to a file?


def get_relevant_events(events: List[Event]) -> List[Event]:
    """Gets only relevant events.

    Reduce the applied events only to relevant events.
    Useful for when trackers are large, as we traverse them
    for each marker separately.
    """
    relevant_events: List[Event] = []
    for e in events:
        if isinstance(e, SlotSet):
            relevant_events.append(e)
        elif isinstance(e, ActionExecuted):
            relevant_events.append(e)
        elif isinstance(e, UserUttered):
            relevant_events.append(e)
    return relevant_events


class Marker:
    """Marker class."""

    def __init__(self, name: Text, operator: Text, condition: List[Dict[Text, List]]):
        """Initialises Marker."""
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
        return list(set(all_items))  # return unique values

    def does_marker_apply(self, events: List[Event]) -> bool:
        """Checks if the marker applied."""
        if self.operator == "AND":
            return self.check_and(events)

        elif self.operator == "OR":
            return self.check_or(events)

        elif self.operator == "SEQ":
            return self.check_seq(events)

        else:
            raise RasaException(
                "Invalid marker operator - '{self.operator}' was given, \
                options 'AND', 'OR', and 'SEQ' exist."
            )

    def check_and(self, events: List[Event]) -> bool:
        """Checks that the AND condition applies."""
        # whether each event occurred in positive conditions
        occurred_pos = dict.fromkeys(
            self.slot_set + self.action_executed + self.intent_detected, False,
        )
        # whether each event occurred in negative conditions
        occurred_neg = dict.fromkeys(
            self.slot_not_set + self.action_not_executed + self.intent_not_detected,
            False,
        )
        # whether the marker condition was satisfied, at least once
        satisfied_at_least_once = False
        # track the number of user turns so far
        user_turn_index = 0

        for e in events:
            if isinstance(e, SlotSet):
                if e.key in self.slot_set and e.value:
                    occurred_pos[e.key] = True
                if e.key in self.slot_not_set and e.value:
                    occurred_neg[e.key] = True

                if self._check_all_conditions(occurred_pos, occurred_neg):
                    satisfied_at_least_once = True
                    self._mark_event(e.timestamp, user_turn_index)

            elif isinstance(e, ActionExecuted):
                if e.action_name in self.action_executed:
                    occurred_pos[e.action_name] = True
                if e.action_name in self.action_not_executed:
                    occurred_neg[e.action_name] = True

                if self._check_all_conditions(occurred_pos, occurred_neg):
                    satisfied_at_least_once = True
                    self._mark_event(e.timestamp, user_turn_index)

            elif isinstance(e, UserUttered):
                user_turn_index += 1
                if e.intent.get("name") in self.intent_detected:
                    occurred_pos[str(e.intent.get("name"))] = True
                if e.intent.get("name") in self.intent_not_detected:
                    occurred_neg[str(e.intent.get("name"))] = True

                if self._check_all_conditions(occurred_pos, occurred_neg):
                    satisfied_at_least_once = True
                    self._mark_event(e.timestamp, user_turn_index)

        return satisfied_at_least_once

    @staticmethod
    def _check_all_conditions(
        satisfied_pos: Dict[Text, bool], satisfied_neg: Dict[Text, bool]
    ) -> bool:
        return all(satisfied_pos.values()) and not any(satisfied_neg.values())

    def check_or(self, events: List[Event]) -> bool:
        """Checks that the OR condition applies."""
        satisfied = False
        user_turn_index = 0
        slot_set = []
        action_executed = []
        intent_detected = []

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

    def check_seq(self, events: List[Event]) -> bool:
        """Checks that the SEQ condition applies."""
        return False

    def _mark_event(self, timestamp: float, user_turn_index: int) -> None:
        self.timestamps.append(timestamp)
        self.preceding_user_turns.append(user_turn_index)
