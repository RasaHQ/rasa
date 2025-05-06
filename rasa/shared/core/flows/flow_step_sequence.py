from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text

from rasa.shared.core.flows.flow_step import FlowStep, step_from_json
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class FlowStepSequence:
    """A Sequence of flow steps."""

    child_steps: List[FlowStep]

    @staticmethod
    def from_json(flow_id: Text, data: List[Dict[Text, Any]]) -> FlowStepSequence:
        """Create a FlowStepSequence object from serialized data

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a StepSequence in a serialized format

        Returns:
            A StepSequence object including its flow step objects.
        """
        flow_steps: List[FlowStep] = [
            step_from_json(flow_id, config) for config in data
        ]

        return FlowStepSequence(child_steps=flow_steps)

    def as_json(self) -> List[Dict[Text, Any]]:
        """Serialize the StepSequence object and contained FlowStep objects

        Returns:
            the FlowStepSequence and its FlowSteps as serialized data
        """
        return [
            step.as_json()
            for step in self.child_steps
            if not isinstance(step, InternalFlowStep)
        ]

    def _resolve_steps(self, should_resolve_calls: bool) -> List[FlowStep]:
        """Resolves the steps of the flow."""
        return [
            step
            for child_step in self.child_steps
            for step in child_step.steps_in_tree(
                should_resolve_calls=should_resolve_calls
            )
        ]

    @property
    def steps_with_calls_resolved(self) -> List[FlowStep]:
        """Return all steps in this step sequence and their sub steps."""
        return self._resolve_steps(should_resolve_calls=True)

    @property
    def steps(self) -> List[FlowStep]:
        """Return the steps of the flow without steps of called flows"""
        return self._resolve_steps(should_resolve_calls=False)

    def first(self) -> Optional[FlowStep]:
        """Return the first step of the sequence."""
        return self.child_steps[0] if self.child_steps else None

    @classmethod
    def empty(cls) -> FlowStepSequence:
        """Create an empty FlowStepSequence object."""
        return cls(child_steps=[])

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FlowStepSequence)
            and self.child_steps == other.child_steps
        )
