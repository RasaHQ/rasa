from __future__ import annotations

from typing import Dict, Text, Any

from rasa.shared.core.flows.flow_step import FlowStep


class InternalFlowStep(FlowStep):
    """Represents the configuration of a built-in flow step.

    Built in flow steps are required to manage the lifecycle of a
    flow and are not intended to be used by users.
    """

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> InternalFlowStep:
        """Used to read flow steps from parsed JSON.

        Args:
            flow_step_config: The parsed JSON as a dictionary.

        Returns:
            The parsed flow step.
        """
        raise ValueError("A start step cannot be parsed.")

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        raise ValueError("A start step cannot be dumped.")
