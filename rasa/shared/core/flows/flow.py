from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Set,
    Text,
    Union,
)
import structlog

from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX, UTTER_PREFIX
from rasa.shared.exceptions import RasaException

import rasa.shared.utils.io
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.flows.flow_step import FlowStep

from rasa.shared.core.flows.flow_step_links import StaticFlowStepLink
from rasa.shared.core.flows.nlu_trigger import NLUTriggers
from rasa.shared.core.flows.steps.continuation import ContinueFlowStep
from rasa.shared.core.flows.steps.constants import (
    CONTINUE_STEP_PREFIX,
    START_STEP,
    END_STEP,
)
from rasa.shared.core.flows.steps import (
    CollectInformationFlowStep,
    EndFlowStep,
    LinkFlowStep,
    StartFlowStep,
)
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence


structlogger = structlog.get_logger()


@dataclass
class Flow:
    """Represents the configuration of a flow."""

    id: Text
    """The id of the flow."""
    custom_name: Optional[Text] = None
    """The human-readable name of the flow."""
    description: Optional[Text] = None
    """The description of the flow."""
    guard_condition: Optional[Text] = None
    """The condition that needs to be fulfilled for the flow to be startable."""
    step_sequence: FlowStepSequence = field(default_factory=FlowStepSequence.empty)
    """The steps of the flow."""
    nlu_triggers: Optional[NLUTriggers] = None
    """The list of intents, e.g. nlu triggers, that start the flow."""

    @staticmethod
    def from_json(flow_id: Text, data: Dict[Text, Any]) -> Flow:
        """Create a Flow object from serialized data

        Args:
            data: data for a Flow object in a serialized format.

        Returns:
            A Flow object.
        """
        step_sequence = FlowStepSequence.from_json(data.get("steps"))
        nlu_triggers = NLUTriggers.from_json(data.get("nlu_trigger"))

        return Flow(
            id=flow_id,
            custom_name=data.get("name"),
            description=data.get("description"),
            # str or bool are permitted in the flow schema but internally we want a str
            guard_condition=str(data.get("if")) if data.get("if") else None,
            step_sequence=Flow.resolve_default_ids(step_sequence),
            nlu_triggers=nlu_triggers,
        )

    @staticmethod
    def create_default_name(flow_id: str) -> str:
        """Create a default flow name for when it is missing."""
        return flow_id.replace("_", " ").replace("-", " ")

    @staticmethod
    def resolve_default_ids(step_sequence: FlowStepSequence) -> FlowStepSequence:
        """Resolves the default ids of all steps in the sequence.

        If a step does not have an id, a default id is assigned to it based
        on the type of the step and its position in the flow.

        Similarly, if a step doesn't have an explicit next assigned we resolve
        the default next step id.

        Args:
            step_sequence: The step sequence to resolve the default ids for.

        Returns:
            The step sequence with the default ids resolved.
        """
        # assign an index to all steps
        for idx, step in enumerate(step_sequence.steps):
            step.idx = idx

        def resolve_default_next(steps: List[FlowStep], is_root_sequence: bool) -> None:
            for i, step in enumerate(steps):
                if step.next.no_link_available():
                    if i == len(steps) - 1:
                        # can't attach end to link step
                        if is_root_sequence and not isinstance(step, LinkFlowStep):
                            # if this is the root sequence, we need to add an end step
                            # to the end of the sequence. other sequences, e.g.
                            # in branches need to explicitly add a next step.
                            step.next.links.append(StaticFlowStepLink(END_STEP))
                    else:
                        step.next.links.append(StaticFlowStepLink(steps[i + 1].id))
                for link in step.next.links:
                    if sub_steps := link.child_steps():
                        resolve_default_next(sub_steps, is_root_sequence=False)

        resolve_default_next(step_sequence.child_steps, is_root_sequence=True)
        return step_sequence

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the Flow object.

        Returns:
            The Flow object as serialized data.
        """
        data = {
            "id": self.id,
            "steps": self.step_sequence.as_json(),
        }
        if self.custom_name is not None:
            data["name"] = self.custom_name
        if self.description is not None:
            data["description"] = self.description
        if self.guard_condition is not None:
            data["if"] = self.guard_condition
        if self.nlu_triggers:
            data["nlu_trigger"] = self.nlu_triggers.as_json()

        return data

    def readable_name(self) -> str:
        """Returns the name of the flow or its id if no name is set."""
        return self.name or self.id

    def step_by_id(self, step_id: Optional[Text]) -> Optional[FlowStep]:
        """Returns the step with the given id."""
        if not step_id:
            return None

        if step_id == START_STEP:
            return StartFlowStep(self.first_step_in_flow().id)

        if step_id == END_STEP:
            return EndFlowStep()

        if step_id.startswith(CONTINUE_STEP_PREFIX):
            return ContinueFlowStep(step_id[len(CONTINUE_STEP_PREFIX) :])

        for step in self.steps:
            if step.id == step_id:
                return step

        return None

    def first_step_in_flow(self) -> FlowStep:
        """Returns the start step of this flow."""
        if len(self.steps) == 0:
            raise RuntimeError(
                f"Flow {self.id} is empty despite validation that this cannot happen."
            )
        return self.steps[0]

    def previous_collect_steps(
        self, step_id: Optional[str]
    ) -> List[CollectInformationFlowStep]:
        """Return the CollectInformationFlowSteps asked before the given step.

        CollectInformationFlowSteps are returned roughly in reverse order,
        i.e. the first step in the list is the one that was asked last. However,
        due to circles in the flow, the order is not guaranteed to be exactly reverse.
        """

        def _previously_asked_collect(
            current_step_id: str, visited_steps: Set[str]
        ) -> List[CollectInformationFlowStep]:
            """Returns the collect information steps asked before the given step.

            Keeps track of the steps that have been visited to avoid circles.
            """
            current_step = self.step_by_id(current_step_id)

            collects: List[CollectInformationFlowStep] = []

            if not current_step:
                return collects

            if isinstance(current_step, CollectInformationFlowStep):
                collects.append(current_step)

            visited_steps.add(current_step.id)

            for previous_step in self.steps:
                for next_link in previous_step.next.links:
                    if next_link.target != current_step_id:
                        continue
                    if previous_step.id in visited_steps:
                        continue
                    collects.extend(
                        _previously_asked_collect(previous_step.id, visited_steps)
                    )
            return collects

        return _previously_asked_collect(step_id or START_STEP, set())

    @property
    def is_rasa_default_flow(self) -> bool:
        """Test whether the flow is a rasa default flow."""
        return self.id.startswith(RASA_DEFAULT_FLOW_PATTERN_PREFIX)

    def get_collect_steps(self) -> List[CollectInformationFlowStep]:
        """Return all CollectInformationFlowSteps in the flow."""
        collect_steps = []
        for step in self.steps:
            if isinstance(step, CollectInformationFlowStep):
                collect_steps.append(step)
        return collect_steps

    @property
    def steps(self) -> List[FlowStep]:
        """Return the steps of the flow."""
        return self.step_sequence.steps

    @cached_property
    def fingerprint(self) -> str:
        """Create a fingerprint identifying this step sequence."""
        return rasa.shared.utils.io.deep_container_fingerprint(self.as_json())

    @property
    def utterances(self) -> Set[str]:
        """Retrieve all utterances of this flow"""
        return set().union(*[step.utterances for step in self.step_sequence.steps])

    @property
    def name(self) -> str:
        """Create a default name if none is present."""
        return self.custom_name or Flow.create_default_name(self.id)

    def first(self) -> Optional[FlowStep]:
        """Returns the first step of the sequence."""
        if len(self.child_steps) == 0:
            return None
        return self.child_steps[0]


def step_from_json(flow_step_config: Dict[Text, Any]) -> FlowStep:
    """Used to read flow steps from parsed YAML.

    Args:
        flow_step_config: The parsed YAML as a dictionary.

    Returns:
        The parsed flow step.
    """
    if "action" in flow_step_config:
        return ActionFlowStep.from_json(flow_step_config)
    if "collect" in flow_step_config:
        return CollectInformationFlowStep.from_json(flow_step_config)
    if "link" in flow_step_config:
        return LinkFlowStep.from_json(flow_step_config)
    if "set_slots" in flow_step_config:
        return SetSlotsFlowStep.from_json(flow_step_config)
    if "generation_prompt" in flow_step_config:
        return GenerateResponseFlowStep.from_json(flow_step_config)
    else:
        return BranchFlowStep.from_json(flow_step_config)


@dataclass
class FlowStep:
    """Represents the configuration of a flow step."""

    custom_id: Optional[Text]
    """The id of the flow step."""
    idx: int
    """The index of the step in the flow."""
    description: Optional[Text]
    """The description of the flow step."""
    metadata: Dict[Text, Any]
    """Additional, unstructured information about this flow step."""
    next: "FlowLinks"
    """The next steps of the flow step."""

    @classmethod
    def _from_json(cls, flow_step_config: Dict[Text, Any]) -> FlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            data: The context and slots to evaluate the start condition against.

        Returns:
            Whether the start condition is satisfied.
        """
        return FlowStep(
            # the idx is set later once the flow is created that contains
            # this step
            idx=-1,
            custom_id=flow_step_config.get("id"),
            description=flow_step_config.get("description"),
            metadata=flow_step_config.get("metadata", {}),
            next=FlowLinks.from_json(flow_step_config.get("next", [])),
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = {"next": self.next.as_json(), "id": self.id}

        if self.description:
            dump["description"] = self.description
        if self.metadata:
            dump["metadata"] = self.metadata
        return dump

    def steps_in_tree(self) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow step."""
        yield self
        yield from self.next.steps_in_tree()

    @property
    def id(self) -> Text:
        """Returns the id of the flow step."""
        return self.custom_id or self.default_id()

    def default_id(self) -> str:
        """Returns the default id of the flow step."""
        return f"{self.idx}_{self.default_id_postfix()}"

    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        raise NotImplementedError()

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step"""
        return set()


class InternalFlowStep(FlowStep):
    """Represents the configuration of a built-in flow step.

    Built in flow steps are required to manage the lifecycle of a
    flow and are not intended to be used by users.
    """

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> ActionFlowStep:
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


@dataclass
class StartFlowStep(InternalFlowStep):
    """Represents the configuration of a start flow step."""

    def __init__(self, start_step_id: Optional[Text]) -> None:
        """Initializes a start flow step.

        Args:
            start_step: The step to start the flow from.
        """
        if start_step_id is not None:
            links: List[FlowLink] = [StaticFlowLink(target=start_step_id)]
        else:
            links = []

        super().__init__(
            idx=0,
            custom_id=START_STEP,
            description=None,
            metadata={},
            next=FlowLinks(links=links),
        )


@dataclass
class EndFlowStep(InternalFlowStep):
    """Represents the configuration of an end to a flow."""

    def __init__(self) -> None:
        """Initializes an end flow step."""
        super().__init__(
            idx=0,
            custom_id=END_STEP,
            description=None,
            metadata={},
            next=FlowLinks(links=[]),
        )


CONTINUE_STEP_PREFIX = "NEXT:"


@dataclass
class ContinueFlowStep(InternalFlowStep):
    """Represents the configuration of a continue-step flow step."""

    def __init__(self, next: str) -> None:
        """Initializes a continue-step flow step."""
        super().__init__(
            idx=0,
            custom_id=CONTINUE_STEP_PREFIX + next,
            description=None,
            metadata={},
            # The continue step links to the step that should be continued.
            # The flow policy in a sense only "runs" the logic of a step
            # when it transitions to that step, once it is there it will use
            # the next link to transition to the next step. This means that
            # if we want to "re-run" a step, we need to link to it again.
            # This is why the continue step links to the step that should be
            # continued.
            next=FlowLinks(links=[StaticFlowLink(target=next)]),
        )

    @staticmethod
    def continue_step_for_id(step_id: str) -> str:
        return CONTINUE_STEP_PREFIX + step_id


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
            action=flow_step_config.get("action", ""),
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

    def default_id_postfix(self) -> str:
        return self.action

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step"""
        return {self.action} if self.action.startswith(UTTER_PREFIX) else set()


@dataclass
class BranchFlowStep(FlowStep):
    """Represents the configuration of a branch flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> BranchFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return BranchFlowStep(**base.__dict__)

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        return dump

    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "branch"


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
            link=flow_step_config.get("link", ""),
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

    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"link_{self.link}"


DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 5,
    "temperature": DEFAULT_OPENAI_TEMPERATURE,
    "model_name": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
}


@dataclass
class GenerateResponseFlowStep(FlowStep):
    """Represents the configuration of a step prompting an LLM."""

    generation_prompt: Text
    """The prompt template of the flow step."""
    llm_config: Optional[Dict[Text, Any]] = None
    """The LLM configuration of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> GenerateResponseFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return GenerateResponseFlowStep(
            generation_prompt=flow_step_config.get("generation_prompt", ""),
            llm_config=flow_step_config.get("llm", None),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["generation_prompt"] = self.generation_prompt
        if self.llm_config:
            dump["llm"] = self.llm_config

        return dump

    def generate(self, tracker: DialogueStateTracker) -> Optional[Text]:
        """Generates a response for the given tracker.

        Args:
            tracker: The tracker to generate a response for.

        Returns:
            The generated response.
        """
        from rasa.shared.utils.llm import llm_factory, tracker_as_readable_transcript
        from jinja2 import Template

        context = {
            "history": tracker_as_readable_transcript(tracker, max_turns=5),
            "latest_user_message": tracker.latest_message.text
            if tracker.latest_message
            else "",
        }
        context.update(tracker.current_slot_values())

        llm = llm_factory(self.llm_config, DEFAULT_LLM_CONFIG)
        prompt = Template(self.generation_prompt).render(context)

        try:
            predicate = Predicate(self.guard_condition)
            is_startable = predicate.evaluate(data)
            structlogger.debug(
                "command_generator.validate_flow_starting_conditions.result",
                predicate=predicate.description(),
                is_startable=is_startable,
            )
            return is_startable
        except (TypeError, Exception) as e:
            structlogger.error(
                "command_generator.validate_flow_starting_conditions.error",
                predicate=self.guard_condition,
                context=data,
                error=str(e),
            )
            return False
