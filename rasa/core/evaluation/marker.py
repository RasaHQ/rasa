from typing import Text
from rasa.core.evaluation.marker_base import (
    CompoundMarker,
    AtomicMarker,
    configurable_via,
)
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event
from rasa.shared.nlu.constants import INTENT_NAME_KEY

# TODO: constants


@configurable_via(tag="and", negated_tag="one_not")
class AndMarker(CompoundMarker):
    """Checks that all sub-markers apply."""

    def _to_str_with(self, tag: Text) -> Text:
        marker_str = f" {tag} ".join(str(marker) for marker in self.sub_markers)
        return "({})".format(marker_str)

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return all(marker.history[-1] for marker in self.sub_markers)


@configurable_via(tag="or", negated_tag="not_any")
class OrMarker(CompoundMarker):
    """Checks that one sub-markers is applies."""

    def _to_str_with(self, tag: Text) -> Text:
        marker_str = f" {tag} ".join(str(marker) for marker in self.sub_markers)
        return "({})".format(marker_str)

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return any(marker.history[-1] for marker in self.sub_markers)


@configurable_via(tag="not")
class NotAnyMarker(CompoundMarker):
    """Checks that none of the sub-markers applies."""

    def _to_str_with(self, tag: Text) -> Text:
        sub_markers_str = " or ".join(str(marker) for marker in self.sub_markers)
        return f"{tag}({sub_markers_str})"

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return not any(marker.history[-1] for marker in self.sub_markers)


@configurable_via(tag="seq")
class SequenceMarker(CompoundMarker):
    """Checks the sub-markers applied in the given order."""

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


@configurable_via(tag="action_executed", negated_tag="action_not_executed")
class ActionExecutedMarker(AtomicMarker):
    """Checks whether an action is executed at the current step."""

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return isinstance(event, ActionExecuted) and event.action_name == self.text


@configurable_via(tag="intent_detected", negated_tag="intent_not_detected")
class IntentDetectedMarker(AtomicMarker):
    """Checks whether an intent is expressed at the current step."""

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        return (
            isinstance(event, UserUttered)
            and event.intent.get(INTENT_NAME_KEY) == self.text
        )


@configurable_via(tag="slot_set", negated_tag="slot_not_set")
class SlotSetMarker(AtomicMarker):
    """Checks whether a slot is set at the current step.

    The actual `SlotSet` event might have happened at an earlier step.
    """

    def _non_negated_version_applies_at(self, event: Event) -> bool:
        if isinstance(event, SlotSet) and event.key == self.text:
            # it might be un-set
            return event.value is not None
        else:
            # it is still set
            return bool(len(self.history) and self.history[-1])
