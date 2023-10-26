from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import List, Union, Dict, Text, Any, Optional, Generator

from rasa.shared.core.flows.flow_step import FlowStep

if TYPE_CHECKING:
    from rasa.shared.core.flows.flow_step_sequence import StepSequence


@dataclass
class FlowLinks:
    """Represents the configuration of a list of flow links."""

    links: List[FlowLink]

    @staticmethod
    def from_json(flow_links_config: Union[str, List[Dict[Text, Any]]]) -> FlowLinks:
        """Used to read flow links from parsed YAML.

        Args:
            flow_links_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow links.
        """
        if not flow_links_config:
            return FlowLinks(links=[])

        if isinstance(flow_links_config, str):
            return FlowLinks(links=[StaticFlowLink.from_json(flow_links_config)])

        return FlowLinks(
            links=[
                BranchBasedLink.from_json(link_config)
                for link_config in flow_links_config
                if link_config
            ]
        )

    def as_json(self) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """Returns the flow links as a dictionary.

        Returns:
            The flow links as a dictionary.
        """
        if not self.links:
            return None

        if len(self.links) == 1 and isinstance(self.links[0], StaticFlowLink):
            return self.links[0].as_json()

        return [link.as_json() for link in self.links]

    def no_link_available(self) -> bool:
        """Returns whether no link is available."""
        return len(self.links) == 0

    def steps_in_tree(self) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow links."""
        for link in self.links:
            yield from link.steps_in_tree()


class FlowLink:
    """Represents a flow link."""

    @property
    def target(self) -> Optional[Text]:
        """Returns the target of the flow link.

        Returns:
            The target of the flow link.
        """
        raise NotImplementedError()

    def as_json(self) -> Any:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        raise NotImplementedError()

    @staticmethod
    def from_json(link_config: Any) -> FlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        raise NotImplementedError()

    def steps_in_tree(self) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow link."""
        raise NotImplementedError()

    def child_steps(self) -> List[FlowStep]:
        """Returns the child steps of the flow link."""
        raise NotImplementedError()


@dataclass
class BranchBasedLink(FlowLink):
    target_reference: Union[Text, StepSequence]
    """The id of the linked flow."""

    def steps_in_tree(self) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow link."""
        from rasa.shared.core.flows.flow_step_sequence import StepSequence

        if isinstance(self.target_reference, StepSequence):
            yield from self.target_reference.steps

    def child_steps(self) -> List[FlowStep]:
        """Returns the child steps of the flow link."""
        from rasa.shared.core.flows.flow_step_sequence import StepSequence

        if isinstance(self.target_reference, StepSequence):
            return self.target_reference.child_steps
        else:
            return []

    @property
    def target(self) -> Optional[Text]:
        """Returns the target of the flow link."""
        from rasa.shared.core.flows.flow_step_sequence import StepSequence

        if isinstance(self.target_reference, StepSequence):
            if first := self.target_reference.first():
                return first.id
            else:
                return None
        else:
            return self.target_reference

    @staticmethod
    def from_json(link_config: Dict[Text, Any]) -> BranchBasedLink:
        """Used to read a single flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        if "if" in link_config:
            return IfFlowLink.from_json(link_config)
        else:
            return ElseFlowLink.from_json(link_config)


@dataclass
class IfFlowLink(BranchBasedLink):
    """Represents the configuration of an if flow link."""

    condition: Optional[Text]
    """The condition of the linked flow."""

    @staticmethod
    def from_json(link_config: Dict[Text, Any]) -> IfFlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        from rasa.shared.core.flows.flow_step_sequence import StepSequence

        if isinstance(link_config["then"], str):
            return IfFlowLink(
                target_reference=link_config["then"], condition=link_config.get("if")
            )
        else:
            return IfFlowLink(
                target_reference=StepSequence.from_json(link_config["then"]),
                condition=link_config.get("if"),
            )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        from rasa.shared.core.flows.flow_step_sequence import StepSequence

        return {
            "if": self.condition,
            "then": self.target_reference.as_json()
            if isinstance(self.target_reference, StepSequence)
            else self.target_reference,
        }


@dataclass
class ElseFlowLink(BranchBasedLink):
    """Represents the configuration of an else flow link."""

    @staticmethod
    def from_json(link_config: Dict[Text, Any]) -> ElseFlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        from rasa.shared.core.flows.flow_step_sequence import StepSequence

        if isinstance(link_config["else"], str):
            return ElseFlowLink(target_reference=link_config["else"])
        else:
            return ElseFlowLink(
                target_reference=StepSequence.from_json(link_config["else"])
            )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        from rasa.shared.core.flows.flow_step_sequence import StepSequence

        return {
            "else": self.target_reference.as_json()
            if isinstance(self.target_reference, StepSequence)
            else self.target_reference
        }


@dataclass
class StaticFlowLink(FlowLink):
    """Represents the configuration of a static flow link."""

    target_id: Text
    """The id of the linked flow."""

    @staticmethod
    def from_json(link_config: Text) -> StaticFlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        return StaticFlowLink(link_config)

    def as_json(self) -> Text:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        return self.target

    def steps_in_tree(self) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow link."""
        # static links do not have any child steps
        yield from []

    def child_steps(self) -> List[FlowStep]:
        """Returns the child steps of the flow link."""
        return []

    @property
    def target(self) -> Optional[Text]:
        """Returns the target of the flow link."""
        return self.target_id
