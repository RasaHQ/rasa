from __future__ import annotations

from typing import runtime_checkable, Protocol

from rasa.shared.core.trackers import DialogueStateTracker


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
