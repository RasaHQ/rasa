from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set, Text
from rasa.shared.exceptions import RasaException

import rasa.shared.utils.io


class UnreachableFlowStepException(RasaException):
    """Raised when a flow step is unreachable."""

    def __init__(self, step: FlowStep, flow: Flow) -> None:
        """Initializes the exception."""
        self.step = step
        self.flow = flow

    def __str__(self) -> Text:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step.id}' in flow '{self.flow.id}' can not be reached from the start step. "
            f"Please make sure that all steps can be reached from the start step, e.g. by "
            f"checking that another step points to this step."
        )


class UnresolvedFlowStepIdException(RasaException):
    """Raised when a flow step is referenced but it's id can not be resolved."""

    def __init__(
        self, step_id: Text, flow: Flow, referenced_from: Optional[FlowStep]
    ) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow = flow
        self.referenced_from = referenced_from

    def __str__(self) -> Text:
        """Return a string representation of the exception."""
        if self.referenced_from:
            exception_message = (
                f"Step with id '{self.step_id}' could not be resolved. "
                f"'Step '{self.referenced_from.id}' in flow '{self.flow.id}' "
                f"referenced this step but it does not exist. "
            )
        else:
            exception_message = (
                f"Step '{self.step_id}' in flow '{self.flow.id}' can not be resolved. "
            )

        return exception_message + (
            f"Please make sure that the step is defined in the same flow."
        )


class UnresolvedFlowException(RasaException):
    """Raised when a flow is referenced but it's id can not be resolved."""

    def __init__(self, flow_id: Text) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id

    def __str__(self) -> Text:
        """Return a string representation of the exception."""
        return (
            f"Flow '{self.flow_id}' can not be resolved. "
            f"Please make sure that the flow is defined."
        )


class FlowsList:
    """Represents the configuration of a list of flow.

    We need this class to be able to fingerprint the flows configuration.
    Fingerprinting is needed to make sure that the model is retrained if the
    flows configuration changes.
    """

    def __init__(self, flows: List[Flow]) -> None:
        """Initializes the configuration of flows.

        Args:
            flows: The flows to be configured.
        """
        self.underlying_flows = flows

    def is_empty(self) -> bool:
        """Returns whether the flows list is empty."""
        return len(self.underlying_flows) == 0

    @classmethod
    def from_json(
        cls, flows_configs: Optional[Dict[Text, Dict[Text, Any]]]
    ) -> FlowsList:
        """Used to read flows from parsed YAML.

        Args:
            flows_configs: The parsed YAML as a dictionary.

        Returns:
            The parsed flows.
        """
        if not flows_configs:
            return cls([])

        return cls(
            [
                Flow.from_json(flow_id, flow_config)
                for flow_id, flow_config in flows_configs.items()
            ]
        )

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

    def flow_by_id(self, id: Optional[Text]) -> Optional[Flow]:
        """Return the flow with the given id."""
        if not id:
            return None

        for flow in self.underlying_flows:
            if flow.id == id:
                return flow
        else:
            return None

    def first_step(self, flow_id: Text) -> Optional[FlowStep]:
        """Return the first step of the flow."""
        flow = self.flow_by_id(flow_id)
        return flow.start_step() if flow else None

    def step_by_id(self, step_id: Text, flow_id: Text) -> FlowStep:
        """Return the step with the given id."""
        flow = self.flow_by_id(flow_id)
        if not flow:
            raise UnresolvedFlowException(flow_id)

        step = flow.step_for_id(step_id)
        if not step:
            raise UnresolvedFlowStepIdException(step_id, flow)

        return step

    def validate(self) -> None:
        """Validate the flows."""
        for flow in self.underlying_flows:
            flow.validate()


@dataclass
class Flow:
    """Represents the configuration of a flow."""

    id: Text
    """The id of the flow."""
    name: Text
    """The name of the flow."""
    description: Optional[Text]
    """The description of the flow."""
    steps: List[FlowStep]
    """The steps of the flow."""

    @staticmethod
    def from_json(flow_id: Text, flow_config: Dict[Text, Any]) -> Flow:
        """Used to read flows from parsed YAML.

        Args:
            flow_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow.
        """
        return Flow(
            id=flow_id,
            name=flow_config.get("name"),
            description=flow_config.get("description"),
            steps=[
                step_from_json(step_config)
                for step_config in flow_config.get("steps", [])
            ],
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow as a dictionary.

        Returns:
            The flow as a dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [step.as_json() for step in self.steps],
        }

    def validate(self) -> None:
        """Validates the flow configuration.

        This ensures that the flow semantically makes sense. E.g. it
        checks:
        -"""

        self._validate_all_next_ids_are_availble_steps()
        self._validate_all_steps_can_be_reached()

    def _validate_all_next_ids_are_availble_steps(self) -> None:
        """Validates that all next links point to existing steps."""

        available_steps = {step.id for step in self.steps}
        for step in self.steps:
            for link in step.next.links:
                if link.target not in available_steps:
                    raise UnresolvedFlowStepIdException(link.target, self, step)

    def _validate_all_steps_can_be_reached(self) -> None:
        """Validates that all steps can be reached from the start step."""

        def _reachable_steps(
            step: Optional[FlowStep], reached_steps: Set[Text]
        ) -> Set[FlowStep]:
            """Validates that the given step can be reached from the start step."""
            if step is None or step.id in reached_steps:
                return reached_steps

            reached_steps.add(step.id)
            for link in step.next.links:
                reached_steps = _reachable_steps(
                    self.step_for_id(link.target), reached_steps
                )
            return reached_steps

        reached_steps = _reachable_steps(self.start_step(), set())

        for step in self.steps:
            if step.id not in reached_steps:
                raise UnreachableFlowStepException(step, self)

    def step_for_id(self, step_id: Text) -> Optional[FlowStep]:
        """Returns the step with the given id."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def start_step(self) -> Optional[FlowStep]:
        """Returns the start step of this flow."""
        if len(self.steps) == 0:
            return None
        return self.steps[0]


def step_from_json(flow_step_config: Dict[Text, Any]) -> FlowStep:
    """Used to read flow steps from parsed YAML.

    Args:
        flow_step_config: The parsed YAML as a dictionary.

    Returns:
        The parsed flow step.
    """
    if "action" in flow_step_config:
        return ActionFlowStep.from_json(flow_step_config)
    if "intent" in flow_step_config:
        return IntentFlowStep.from_json(flow_step_config)
    if "user" in flow_step_config:
        return UserFlowStep.from_json(flow_step_config)
    if "question" in flow_step_config:
        return QuestionFlowStep.from_json(flow_step_config)
    if "link" in flow_step_config:
        return LinkFlowStep.from_json(flow_step_config)
    else:
        raise ValueError(f"Flow step is missing a type. {flow_step_config}")


@dataclass
class FlowStep:
    """Represents the configuration of a flow step."""

    id: Text
    """The id of the flow step."""
    next: "FlowLinks"
    """The next steps of the flow step."""

    @classmethod
    def _from_json(cls, flow_step_config: Dict[Text, Any]) -> FlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        return FlowStep(
            id=flow_step_config["id"],
            next=FlowLinks.from_json(flow_step_config.get("next", [])),
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        return {
            "id": self.id,
            "next": self.next.as_json(),
        }

    def has_next(self) -> bool:
        """Returns whether the flow step has a next steps."""
        return bool(self.next.links)


@dataclass
class ActionFlowStep(FlowStep):
    """Represents the configuration of an action flow step."""

    action: Text
    """The action of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> ActionFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return ActionFlowStep(
            action=flow_step_config.get("action"),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["action"] = self.action
        return dump


@dataclass
class LinkFlowStep(FlowStep):
    """Represents the configuration of a link flow step."""

    link: Text
    """The link of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> LinkFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return LinkFlowStep(
            link=flow_step_config.get("link"),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["link"] = self.link
        return dump


@dataclass
class IntentFlowStep(FlowStep):
    """Represents the configuration of an intent flow step."""

    intent: Text
    """The intent of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> IntentFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return IntentFlowStep(
            intent=flow_step_config.get("intent"),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["intent"] = self.intent
        return dump


@dataclass
class QuestionFlowStep(FlowStep):
    """Represents the configuration of a question flow step."""

    question: Text
    """The question of the flow step."""
    reset_on_reentry: bool = False
    """Whether the question should be reset on reentry."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> QuestionFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return QuestionFlowStep(
            question=flow_step_config.get("question"),
            reset_on_reentry=flow_step_config.get("reset_on_reentry", False),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["question"] = self.question
        dump["reset_on_reentry"] = self.reset_on_reentry
        return dump


@dataclass
class UserFlowStep(FlowStep):
    """Represents the configuration of a user flow step."""

    user: Text
    """The user of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> UserFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return UserFlowStep(
            user=flow_step_config.get("user"),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["user"] = self.user
        return dump


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

    @property
    def target(self) -> Text:
        """Returns the target of the flow link.

        Returns:
            The target of the flow link.
        """
        ...

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
        return {
            "if": self.condition,
            "then": self.target,
        }


@dataclass
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


@dataclass
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
