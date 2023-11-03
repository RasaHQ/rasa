from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Text, Optional, Dict, Any, List, Set

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
from rasa.shared.core.flows.steps.end import EndFlowStep
from rasa.shared.core.flows.steps.start import StartFlowStep
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.flows.steps.link import LinkFlowStep
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence


@dataclass
class Flow:
    """Represents the configuration of a flow."""

    id: Text
    """The id of the flow."""
    custom_name: Optional[Text] = None
    """The human-readable name of the flow."""
    description: Optional[Text] = None
    """The description of the flow."""
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

    def get_trigger_intents(self) -> Set[str]:
        """Returns the trigger intents of the flow"""
        results: Set[str] = set()

        if not self.nlu_triggers:
            return results

        for condition in self.nlu_triggers.trigger_conditions:
            results.add(condition.intent)

        return results

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
