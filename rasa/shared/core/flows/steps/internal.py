from __future__ import annotations

from typing import Dict, Text, Any

from rasa.shared.core.flows.flow_step import FlowStep


class InternalFlowStep(FlowStep):
    """A superclass for built-in flow steps.

    Built-in flow steps are required to manage the lifecycle of a
    flow and are not intended to be used by users.
    """

    @classmethod
    def from_json(cls, data: Dict[Text, Any]) -> InternalFlowStep:
        """Create an InternalFlowStep object from serialized data.

        Args:
            data: data for an InternalFlowStep in a serialized format

        Returns:
            Raises because InternalFlowSteps are not serialized or de-serialized.
        """
        raise ValueError(
            "Internal flow steps are ephemeral and are not to be serialized "
            "or de-serialized."
        )

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the InternalFlowStep object

        Returns:
            Raises because InternalFlowSteps are not serialized or de-serialized.
        """
        raise ValueError(
            "Internal flow steps are ephemeral and are not to be serialized "
            "or de-serialized."
        )

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        raise ValueError("Internal flow steps do not need a default id")
