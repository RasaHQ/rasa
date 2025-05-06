from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional, Text

from rasa.shared.core.flows.flow_step import FlowStep

if TYPE_CHECKING:
    from rasa.shared.core.flows.flow import Flow


@dataclass
class CallFlowStep(FlowStep):
    """Represents the configuration of an call flow step."""

    call: Text
    """The flow to be called."""
    called_flow_reference: Optional["Flow"] = None

    @classmethod
    def from_json(cls, flow_id: Text, data: Dict[Text, Any]) -> CallFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_id: The id of the flow that contains the step.
            data: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super().from_json(flow_id, data)
        return CallFlowStep(
            call=data.get("call", ""),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:  # type: ignore[override]
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        return super().as_json(step_properties={"call": self.call})

    def steps_in_tree(
        self, should_resolve_calls: bool = True
    ) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow step."""
        yield self

        if should_resolve_calls:
            if not self.called_flow_reference:
                raise ValueError("Call flow reference not set.")

            yield from self.called_flow_reference.steps_with_calls_resolved

        yield from self.next.steps_in_tree(should_resolve_calls)

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"call_{self.call}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return (
                self.call == other.call
                and self.called_flow_reference == other.called_flow_reference
                and super().__eq__(other)
            )
        return False
