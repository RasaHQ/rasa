from typing import Optional, Text
from rasa.core.evaluation.marker_base import (
    CompoundMarker,
    AtomicMarker,
    configurable_marker,
)
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event
from rasa.shared.nlu.constants import INTENT_NAME_KEY


@configurable_marker
class AndMarker(CompoundMarker):
    """Checks that all sub-markers apply."""

    @classmethod
    def tag(cls) -> Text:
        """Returns the tag to be used in a config file."""
        return "and"

    @classmethod
    def negated_tag(cls) -> Text:
        """Returns the tag to be used in a config file for the negated version."""
        return "at_least_one_not"

    def _to_str_with(self, tag: Text) -> Text:
        marker_str = f" {tag} ".join(str(marker) for marker in self.sub_markers)
        return f"({marker_str})"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return all(marker.history[-1] for marker in self.sub_markers)


@configurable_marker
class OrMarker(CompoundMarker):
    """Checks that at least one sub-marker applies."""

    @classmethod
    def tag(cls) -> Text:
        """Returns the tag to be used in a config file."""
        return "or"

    @classmethod
    def negated_tag(cls) -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "not"

    def _to_str_with(self, tag: Text) -> Text:
        marker_str = f" {tag} ".join(str(marker) for marker in self.sub_markers)
        return f"({marker_str})"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return any(marker.history[-1] for marker in self.sub_markers)


@configurable_marker
class SequenceMarker(CompoundMarker):
    """Checks that all sub-markers applied in the specified order."""

    @classmethod
    def tag(cls) -> Text:
        """Returns the tag to be used in a config file."""
        return "seq"

    def _to_str_with(self, tag: Text) -> Text:
        sub_markers_str = " -> ".join(str(marker) for marker in self.sub_markers)
        return f"{tag}({sub_markers_str})"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        # Note: the sub-markers have been updated before this tracker
        if len(self.history) < len(self.sub_markers) - 1:
            return False
        return all(
            marker.history[-idx - 1]
            for idx, marker in enumerate(reversed(self.sub_markers))
        )


@configurable_marker
class ActionExecutedMarker(AtomicMarker):
    """Checks whether an action is executed at the current step."""

    @classmethod
    def tag(cls) -> Text:
        """Returns the tag to be used in a config file."""
        return "action_executed"

    @classmethod
    def negated_tag(cls) -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "action_not_executed"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return isinstance(event, ActionExecuted) and event.action_name == self.text


@configurable_marker
class IntentDetectedMarker(AtomicMarker):
    """Checks whether an intent is expressed at the current step."""

    @classmethod
    def tag(cls) -> Text:
        """Returns the tag to be used in a config file."""
        return "intent_detected"

    @classmethod
    def negated_tag(cls) -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "intent_not_detected"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return (
            isinstance(event, UserUttered)
            and event.intent.get(INTENT_NAME_KEY) == self.text
        )


@configurable_marker
class SlotSetMarker(AtomicMarker):
    """Checks whether a slot is set at the current step.

    The actual `SlotSet` event might have happened at an earlier step.
    """

    @classmethod
    def tag(cls) -> Text:
        """Returns the tag to be used in a config file."""
        return "slot_is_set"

    @classmethod
    def negated_tag(cls) -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
        return "slot_is_not_set"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        if isinstance(event, SlotSet) and event.key == self.text:
            # it might be un-set
            return event.value is not None
        else:
            # it is still set
            return bool(len(self.history) and self.history[-1])
