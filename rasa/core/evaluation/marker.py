from typing import Optional, Text
from rasa.core.evaluation.marker_base import (
    CompoundMarker,
    AtomicMarker,
    MarkerRegistry,
)
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event


@MarkerRegistry.configurable_marker
class AndMarker(CompoundMarker):
    """Checks that all sub-markers apply."""

    @staticmethod
    def tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "and"

    @staticmethod
    def negated_tag() -> Text:
        """Returns the tag to be used in a config file for the negated version."""
        return "at_least_one_not"

    def _to_str_with(self, tag: Text) -> Text:
        marker_str = f" {tag} ".join(str(marker) for marker in self.sub_markers)
        return f"({marker_str})"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return all(marker.history[-1] for marker in self.sub_markers)


@MarkerRegistry.configurable_marker
class OrMarker(CompoundMarker):
    """Checks that at least one sub-marker applies."""

    @staticmethod
    def tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "or"

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "not"

    def _to_str_with(self, tag: Text) -> Text:
        marker_str = f" {tag} ".join(str(marker) for marker in self.sub_markers)
        return f"({marker_str})"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return any(marker.history[-1] for marker in self.sub_markers)


@MarkerRegistry.configurable_marker
class SequenceMarker(CompoundMarker):
    """Checks that all sub-markers apply to consecutive in the specified order.

    Given a sequence of sub-markers `m_0, m_1,...,m_n`, the sequence marker applies
    at the `i`-th event if sub-marker `m_{n-j}` applies at the `{i-j}`-th event
    for `j` in `[0,..,n]`.
    """

    @staticmethod
    def tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "seq"

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
class ActionExecutedMarker(AtomicMarker):
    """Checks whether an action is executed at the current step."""

    @staticmethod
    def tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "action_executed"

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "action_not_executed"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return isinstance(event, ActionExecuted) and event.action_name == self.text


@MarkerRegistry.configurable_marker
class IntentDetectedMarker(AtomicMarker):
    """Checks whether an intent is expressed at the current step.

    More precisely it applies at an event if this event is a `UserUttered` event
    where either (1) the retrieval intent or (2) just the intent coincides with
    the specified text.
    """

    @staticmethod
    def tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "intent_detected"

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "intent_not_detected"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return isinstance(event, UserUttered) and self.text in [
            event.intent_name,
            event.full_retrieval_intent_name,
        ]


@MarkerRegistry.configurable_marker
class SlotSetMarker(AtomicMarker):
    """Checks whether a slot is set at the current step.

    The actual `SlotSet` event might have happened at an earlier step.
    """

    @staticmethod
    def tag() -> Text:
        """Returns the tag to be used in a config file."""
        return "slot_is_set"

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "slot_is_not_set"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        if isinstance(event, SlotSet) and event.key == self.text:
            # it might be un-set
            return event.value is not None
        else:
            # it is still set
            return bool(len(self.history) and self.history[-1])
