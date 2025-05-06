from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Text, Union

from rasa.shared.core.flows.flow_step import FlowStep

if TYPE_CHECKING:
    from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence


@dataclass
class FlowStepLinks:
    """A list of flow step links."""

    links: List[FlowStepLink]

    @staticmethod
    def from_json(
        flow_id: Text, data: Union[str, List[Dict[Text, Any]]]
    ) -> FlowStepLinks:
        """Create a FlowStepLinks object from a serialized data format.

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a FlowStepLinks object in a serialized format.

        Returns:
            A FlowStepLinks object.
        """
        if not data:
            return FlowStepLinks(links=[])

        if isinstance(data, str):
            return FlowStepLinks(links=[StaticFlowStepLink.from_json(flow_id, data)])

        return FlowStepLinks(
            links=[
                BranchingFlowStepLink.from_json(flow_id, link_config)
                for link_config in data
                if link_config
            ]
        )

    def as_json(self) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """Serialize the FlowStepLinks object.

        Returns:
            The FlowStepLinks object as serialized data.
        """
        if not self.links:
            return None

        if len(self.links) == 1 and isinstance(self.links[0], StaticFlowStepLink):
            return self.links[0].as_json()

        return [link.as_json() for link in self.links]

    def no_link_available(self) -> bool:
        """Returns whether no link is available."""
        return len(self.links) == 0

    def steps_in_tree(
        self, should_resolve_calls: bool = True
    ) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow step links."""
        for link in self.links:
            yield from link.steps_in_tree(should_resolve_calls)

    def depth_in_tree(self) -> int:
        """Returns the max depth in the tree of the flow step links."""
        depth = 0
        for link in self.links:
            depth = max(depth, link.depth_in_tree())
        return depth

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FlowStepLinks):
            return self.links == other.links
        return False


class FlowStepLink:
    """A flow step link that links two steps in a single flow."""

    @property
    def target(self) -> Text:
        """Returns the target flow step id.

        Returns:
            The target flow step id.
        """
        raise NotImplementedError()

    def as_json(self) -> Any:
        """Serialize the FlowStepLink object.

        Returns:
            The FlowStepLink as serialized data.
        """
        raise NotImplementedError()

    @staticmethod
    def from_json(flow_id: Text, data: Any) -> FlowStepLink:
        """Create a FlowStepLink object from a serialized data format.

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a FlowStepLink object in a serialized format.

        Returns:
            The FlowStepLink object.
        """
        raise NotImplementedError()

    def steps_in_tree(
        self, should_resolve_calls: bool = True
    ) -> Generator[FlowStep, None, None]:
        """Recursively generates the steps in the tree."""
        raise NotImplementedError()

    def child_steps(self) -> List[FlowStep]:
        """Returns the steps of the linked FlowStepSequence if any."""
        raise NotImplementedError()

    def depth_in_tree(self) -> int:
        """Returns the depth in the tree."""
        raise NotImplementedError()


@dataclass
class BranchingFlowStepLink(FlowStepLink):
    target_reference: Union[Text, FlowStepSequence]
    """The id of the linked step or a sequence of steps."""

    def steps_in_tree(
        self, should_resolve_calls: bool = True
    ) -> Generator[FlowStep, None, None]:
        """Recursively generates the steps in the tree."""
        from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence

        if isinstance(self.target_reference, FlowStepSequence):
            if should_resolve_calls:
                yield from self.target_reference.steps_with_calls_resolved
            else:
                yield from self.target_reference.steps

    def child_steps(self) -> List[FlowStep]:
        """Returns the steps of the linked flow step sequence if any."""
        from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence

        if isinstance(self.target_reference, FlowStepSequence):
            return self.target_reference.child_steps
        else:
            return []

    @property
    def target(self) -> Text:
        """Return the target flow step id."""
        from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence

        if isinstance(self.target_reference, FlowStepSequence):
            if first := self.target_reference.first():
                return first.id
            else:
                raise RuntimeError(
                    "Step sequence is empty despite previous validation of "
                    "this not happening"
                )
        else:
            return self.target_reference

    @staticmethod
    def from_json(flow_id: Text, data: Dict[Text, Any]) -> BranchingFlowStepLink:
        """Create a BranchingFlowStepLink object from a serialized data format.

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a BranchingFlowStepLink object in a serialized format.

        Returns:
            a BranchingFlowStepLink object.
        """
        if "if" in data:
            return IfFlowStepLink.from_json(flow_id, data)
        else:
            return ElseFlowStepLink.from_json(flow_id, data)

    def depth_in_tree(self) -> int:
        """Returns the depth in the tree."""
        from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence

        if isinstance(self.target_reference, FlowStepSequence):
            depth = 0
            for step in self.target_reference.steps_with_calls_resolved:
                if isinstance(step.next, FlowStepLinks):
                    depth = max(depth, step.next.depth_in_tree())
            return depth + 1
        return 1

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BranchingFlowStepLink):
            return self.target_reference == other.target_reference
        return False


@dataclass
class IfFlowStepLink(BranchingFlowStepLink):
    """A flow step link that links to another step or step sequence conditionally."""

    condition: Text
    """The condition that needs to be satisfied to follow this flow step link."""

    @staticmethod
    def from_json(flow_id: Text, data: Dict[Text, Any]) -> IfFlowStepLink:
        """Create an IfFlowStepLink object from a serialized data format.

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a IfFlowStepLink in a serialized format.

        Returns:
            An IfFlowStepLink object.
        """
        from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence

        if isinstance(data["then"], str):
            return IfFlowStepLink(target_reference=data["then"], condition=data["if"])
        else:
            return IfFlowStepLink(
                target_reference=FlowStepSequence.from_json(flow_id, data["then"]),
                condition=data["if"],
            )

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the IfFlowStepLink object.

        Returns:
            the IfFlowStepLink object as serialized data.
        """
        from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence

        return {
            "if": self.condition,
            "then": self.target_reference.as_json()
            if isinstance(self.target_reference, FlowStepSequence)
            else self.target_reference,
        }


@dataclass
class ElseFlowStepLink(BranchingFlowStepLink):
    """A flow step link that is taken when conditional flow step links weren't taken."""

    @staticmethod
    def from_json(flow_id: Text, data: Dict[Text, Any]) -> ElseFlowStepLink:
        """Create an ElseFlowStepLink object from serialized data.

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for an ElseFlowStepLink in a serialized format

        Returns:
            An ElseFlowStepLink
        """
        from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence

        if isinstance(data["else"], str):
            return ElseFlowStepLink(target_reference=data["else"])
        else:
            return ElseFlowStepLink(
                target_reference=FlowStepSequence.from_json(flow_id, data["else"])
            )

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the ElseFlowStepLink object

        Returns:
            The ElseFlowStepLink as serialized data.
        """
        from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence

        return {
            "else": self.target_reference.as_json()
            if isinstance(self.target_reference, FlowStepSequence)
            else self.target_reference
        }


@dataclass
class StaticFlowStepLink(FlowStepLink):
    """A static flow step link, linking to a step in the same flow unconditionally."""

    target_step_id: Text
    """The id of the linked step."""

    @staticmethod
    def from_json(flow_id: Text, data: Text) -> StaticFlowStepLink:
        """Create a StaticFlowStepLink from serialized data

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a StaticFlowStepLink in a serialized format

        Returns:
            A StaticFlowStepLink object
        """
        return StaticFlowStepLink(data)

    def as_json(self) -> Text:
        """Serialize the StaticFlowStepLink object

        Returns:
            The StaticFlowStepLink object as serialized data.
        """
        return self.target

    def steps_in_tree(
        self, should_resolve_calls: bool = True
    ) -> Generator[FlowStep, None, None]:
        """Recursively generates the steps in the tree."""
        # static links do not have any child steps
        yield from []

    def child_steps(self) -> List[FlowStep]:
        """Returns the steps of the linked FlowStepSequence if any."""
        return []

    @property
    def target(self) -> Text:
        """Returns the target step id."""
        return self.target_step_id

    def depth_in_tree(self) -> int:
        """Returns the depth in the tree."""
        return 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StaticFlowStepLink):
            return self.target_step_id == other.target_step_id
        return False
