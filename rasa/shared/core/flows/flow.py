from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Text

import rasa.shared.utils.io


class FlowsList:
    """Represents the configuration of a list of flow.

    We need this class to be able to fingerprint the flows configuration.
    Fingerprinting is needed to make sure that the model is retrained if the
    flows configuration changes.
    """

    def __init__(self, flows: List["Flow"]) -> None:
        """Initializes the configuration of flows.

        Args:
            flows: The flows to be configured.
        """
        self.underlying_flows = flows

    @classmethod
    def from_json(
        cls, flows_configs: Optional[List[Dict[Text, Dict[Text, Any]]]]
    ) -> FlowsList:
        """Used to read flows from parsed YAML.

        Args:
            flows_configs: The parsed YAML as a dictionary.

        Returns:
            The parsed flows.
        """
        if not flows_configs:
            return cls([])

        return cls([Flow.from_json(flow_config) for flow_config in flows_configs])

    def as_json(self) -> List[Dict[Text, Any]]:
        """Returns the flows as a dictionary.

        Returns:
            The flows as a dictionary.
        """
        return [flow.as_json() for flow in self.underlying_flows]

    def fingerprint(self) -> str:
        """Creates a fingerprint of the flows configuration.

        Returns:
            The fingerprint of the flows configuration.
        """
        flow_dicts = [flow.as_json() for flow in self.underlying_flows]
        return rasa.shared.utils.io.get_list_fingerprint(flow_dicts)

    def merge(self, other: FlowsList) -> FlowsList:
        """Merges two lists of flows together."""
        return FlowsList(self.underlying_flows + other.underlying_flows)


@dataclass
class Flow:
    """Represents the configuration of a flow."""

    flow: Text
    """The name of the flow."""
    id: Text
    """The id of the flow."""
    description: Optional[Text]
    """The description of the flow."""
    steps: List[FlowStep]

    @staticmethod
    def from_json(flow_config: Dict[Text, Any]) -> Flow:
        """Used to read flows from parsed YAML.

        Args:
            flow_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow.
        """
        return Flow(
            flow=flow_config["flow"],
            id=flow_config["id"],
            description=flow_config.get("description"),
            steps=[
                FlowStep.from_json(step_config)
                for step_config in flow_config.get("steps", [])
            ],
        )


@dataclass
class FlowStep:
    """Represents the configuration of a flow step."""

    id: Text
    """The id of the flow step."""
    action: Optional[Text]
    """The action of the flow step."""
    intent: Optional[Text]
    """The intent of the flow step."""
    user: Optional[Text]
    """The user of the flow step."""
    next: "FlowLinks"

    @staticmethod
    def from_json(flow_step_config: Dict[Text, Any]) -> FlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        return FlowStep(
            id=flow_step_config["id"],
            action=flow_step_config.get("action"),
            intent=flow_step_config.get("intent"),
            user=flow_step_config.get("user"),
            next=FlowLinks.from_json(flow_step_config.get("next", [])),
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        return {
            "id": self.id,
            "action": self.action,
            "intent": self.intent,
            "user": self.user,
            "next": self.links.as_json(),
        }


@dataclass
class FlowLinks:
    """Represents the configuration of a list of flow links."""

    links: List[FlowLink]

    @staticmethod
    def from_json(flow_links_config: List[Dict[Text, Any]]) -> FlowLinks:
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
                FlowLinks.link_from_json(link_config)
                for link_config in flow_links_config
                if link_config
            ]
        )

    @staticmethod
    def link_from_json(link_config: Dict[Text, Any]) -> FlowLink:
        """Used to read a single flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        if "if" in link_config:
            return IfFlowLink.from_json(link_config)
        elif "else" in link_config:
            return ElseFlowLink.from_json(link_config)
        else:
            raise Exception("Invalid flow link")

    def as_json(self) -> Any:
        """Returns the flow links as a dictionary.

        Returns:
            The flow links as a dictionary.
        """
        if not self.links:
            return None

        if len(self.links) == 1 and isinstance(self.links[0], StaticFlowLink):
            return self.links[0].as_json()

        return [link.as_json() for link in self.links]


class FlowLink(Protocol):
    """Represents a flow link."""

    def as_json(self) -> Any:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        ...

    @staticmethod
    def from_json(link_config: Dict[Text, Any]) -> FlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        ...


@dataclass
class IfFlowLink:
    """Represents the configuration of an if flow link."""

    target: Text
    """The id of the linked flow."""
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
        return IfFlowLink(target=link_config["then"], condition=link_config.get("if"))

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        return {"then": self.target, "if": self.condition}


class ElseFlowLink:
    """Represents the configuration of an else flow link."""

    target: Text
    """The id of the linked flow."""

    @staticmethod
    def from_json(link_config: Dict[Text, Any]) -> ElseFlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        return ElseFlowLink(target=link_config["else"])

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        return {"else": self.target}


class StaticFlowLink:
    """Represents the configuration of a static flow link."""

    target: Text
    """The id of the linked flow."""

    @staticmethod
    def from_json(link_config: Text) -> StaticFlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        return StaticFlowLink(target=link_config)

    def as_json(self) -> Text:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        return self.target
