from rasa.shared.core.domain import Domain
from typing import Optional, Text, List
from rasa.core.evaluation.marker_base import (
    OperatorMarker,
    ConditionMarker,
    MarkerRegistry,
)
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event
import rasa.shared.utils.io


@MarkerRegistry.configurable_marker
class AndMarker(OperatorMarker):
    """Checks that all sub-markers apply."""

    @staticmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "and"

    @staticmethod
    def negated_tag() -> Text:
        """Returns the tag to be used in a config file for the negated version."""
        return "at_least_one_not"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return all(marker.history[-1] for marker in self.sub_markers)


@MarkerRegistry.configurable_marker
class OrMarker(OperatorMarker):
    """Checks that at least one sub-marker applies."""

    @staticmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "or"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return any(marker.history[-1] for marker in self.sub_markers)


@MarkerRegistry.configurable_marker
class NotMarker(OperatorMarker):
    """Checks that at least one sub-marker applies."""

    @staticmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "not"

    @staticmethod
    def expected_number_of_sub_markers() -> Optional[int]:
        """Returns the expected number of sub-markers (if there is any)."""
        return 1

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return not self.sub_markers[0].history[-1]


@MarkerRegistry.configurable_marker
class SequenceMarker(OperatorMarker):
    """Checks that all sub-markers apply consecutively in the specified order.

    Given a sequence of sub-markers `m_0, m_1,...,m_n`, the sequence marker applies
    at the `i`-th event if sub-marker `m_{n-j}` applies at the `{i-j}`-th event
    for `j` in `[0,..,n]`.
    """

    @staticmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "seq"

    @staticmethod
    def negated_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "not(seq)"

    def _to_str_with(self, tag: Text) -> Text:
        sub_markers_str = " -> ".join(str(marker) for marker in self.sub_markers)
        return f"{tag}({sub_markers_str})"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        # Remember that all the sub-markers have been updated before this tracker.
        # This means the history of the sub-markers already includes a result
        # for the event we evaluate here:
        num_tracked_events_including_current = len(self.sub_markers[0].history)
        if num_tracked_events_including_current < len(self.sub_markers):
            return False
        return all(
            marker.history[-idx - 1]
            for idx, marker in enumerate(reversed(self.sub_markers))
        )


@MarkerRegistry.configurable_marker
class OccurrenceMarker(OperatorMarker):
    """Checks that all sub-markers applied at least once in history.

    It doesn't matter if the sub markers stop applying later in history. If they
    applied at least once they will always evaluate to `True`.
    """

    @staticmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "at_least_once"

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "never"

    @staticmethod
    def expected_number_of_sub_markers() -> Optional[int]:
        """Returns the expected number of sub-markers (if there is any)."""
        return 1

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        if self.history:
            if self.negated:
                occurred_before = not self.history[-1]
            else:
                occurred_before = self.history[-1]
        else:
            occurred_before = False
        return occurred_before or self.sub_markers[0].history[-1]

    def relevant_events(self) -> List[int]:
        """Only return index of first match (see parent class for full docstring)."""
        try:
            return [self.history.index(True)]
        except ValueError:
            return []


@MarkerRegistry.configurable_marker
class ActionExecutedMarker(ConditionMarker):
    """Checks whether an action is executed at the current step."""

    @staticmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "action_executed"

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "action_not_executed"

    def validate_against_domain(self, domain: Domain) -> bool:
        """Checks that this marker (and its children) refer to entries in the domain.
        
        Args:
            domain: The domain to check against
        """
        valid = self.text in domain.action_names_or_texts

        if not valid:
            rasa.shared.utils.io.raise_warning(
                f"Referenced action '{self.text}' does not exist in the domain"
            )

        return valid

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return isinstance(event, ActionExecuted) and event.action_name == self.text


@MarkerRegistry.configurable_marker
class IntentDetectedMarker(ConditionMarker):
    """Checks whether an intent is expressed at the current step.

    More precisely it applies at an event if this event is a `UserUttered` event
    where either (1) the retrieval intent or (2) just the intent coincides with
    the specified text.
    """

    @staticmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "intent_detected"

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "intent_not_detected"

    def validate_against_domain(self, domain: Domain) -> bool:
        """Checks that this marker (and its children) refer to entries in the domain.
        
        Args:
            domain: The domain to check against
        """
        valid = self.text in domain.intent_properties

        if not valid:
            rasa.shared.utils.io.raise_warning(
                f"Referenced intent '{self.text}' does not exist in the domain"
            )

        return valid

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return isinstance(event, UserUttered) and self.text in [
            event.intent_name,
            event.full_retrieval_intent_name,
        ]


@MarkerRegistry.configurable_marker
class SlotSetMarker(ConditionMarker):
    """Checks whether a slot is set at the current step.

    The actual `SlotSet` event might have happened at an earlier step.
    """

    @staticmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "slot_is_set"

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "slot_is_not_set"

    def validate_against_domain(self, domain: Domain) -> bool:
        """Checks that this marker (and its children) refer to entries in the domain.
        
        Args:
            domain: The domain to check against
        """
        valid = any(self.text == slot.name for slot in domain.slots)

        if not valid:
            rasa.shared.utils.io.raise_warning(
                f"Referenced slot '{self.text}' does not exist in the domain"
            )

        return valid

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        if isinstance(event, SlotSet) and event.key == self.text:
            # slot is set if and only if it's value is not `None`
            return event.value is not None
        if self.history:
            was_set = self.history[-1] if not self.negated else not self.history[-1]
            return was_set
        return False
