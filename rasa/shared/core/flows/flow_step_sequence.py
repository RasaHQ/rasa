from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Text, Any, Optional

from rasa.shared.core.flows.flow_step import FlowStep, step_from_json
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class StepSequence:
    child_steps: List[FlowStep]

    @staticmethod
    def from_json(steps_config: List[Dict[Text, Any]]) -> StepSequence:
        """Used to read steps from parsed YAML.

        Args:
            steps_config: The parsed YAML as a dictionary.

        Returns:
            The parsed steps.
        """

        flow_steps: List[FlowStep] = [step_from_json(config) for config in steps_config]

        return StepSequence(child_steps=flow_steps)

    def as_json(self) -> List[Dict[Text, Any]]:
        """Returns the steps as a dictionary.

        Returns:
            The steps as a dictionary.
        """
        return [
            step.as_json()
            for step in self.child_steps
            if not isinstance(step, InternalFlowStep)
        ]

    @property
    def steps(self) -> List[FlowStep]:
        """Returns the steps of the flow."""
        return [
            step
            for child_step in self.child_steps
            for step in child_step.steps_in_tree()
        ]

    def first(self) -> Optional[FlowStep]:
        """Returns the first step of the sequence."""
        if len(self.child_steps) == 0:
            return None
        return self.child_steps[0]
