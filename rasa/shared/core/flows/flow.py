from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Text, Optional, Dict, Any, List, Set
from pypred import Predicate
import structlog

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
    StartFlowStep,
    ActionFlowStep,
)
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.slots import Slot


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
    always_include_in_prompt: Optional[bool] = None
    """
    A flag that checks whether the flow should always be included in the prompt or not.
    """

    @staticmethod
    def from_json(flow_id: Text, data: Dict[Text, Any]) -> Flow:
        """Create a Flow object from serialized data.

        Args:
            flow_id: id of the flow
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
            always_include_in_prompt=data.get("always_include_in_prompt"),
            # str or bool are permitted in the flow schema but internally we want a str
            guard_condition=str(data["if"]) if "if" in data else None,
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
                if step.next.no_link_available() and step.does_allow_for_next_step():
                    if i == len(steps) - 1:
                        if is_root_sequence:
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
        data: Dict[Text, Any] = {
            "id": self.id,
            "steps": self.step_sequence.as_json(),
        }
        if self.custom_name is not None:
            data["name"] = self.custom_name
        if self.description is not None:
            data["description"] = self.description
        if self.guard_condition is not None:
            data["if"] = self.guard_condition
        if self.always_include_in_prompt is not None:
            data["always_include_in_prompt"] = self.always_include_in_prompt
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

        for step in self.steps_with_calls_resolved:
            if step.id == step_id:
                return step

        return None

    def first_step_in_flow(self) -> FlowStep:
        """Returns the start step of this flow."""
        if not (steps := self.steps):
            raise RuntimeError(
                f"Flow {self.id} is empty despite validation that this cannot happen."
            )
        return steps[0]

    def get_trigger_intents(self) -> Set[str]:
        """Returns the trigger intents of the flow."""
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
        for step in self.steps_with_calls_resolved:
            if isinstance(step, CollectInformationFlowStep):
                collect_steps.append(step)
        return collect_steps

    @property
    def steps_with_calls_resolved(self) -> List[FlowStep]:
        """Return the steps of the flow including steps of called flows."""
        return self.step_sequence.steps_with_calls_resolved

    @property
    def steps(self) -> List[FlowStep]:
        """Return the steps of the flow without steps of called flows."""
        return self.step_sequence.steps

    @cached_property
    def fingerprint(self) -> str:
        """Create a fingerprint identifying this step sequence."""
        return rasa.shared.utils.io.deep_container_fingerprint(self.as_json())

    @property
    def utterances(self) -> Set[str]:
        """Retrieve all utterances of this flow."""
        return set().union(
            *[step.utterances for step in self.step_sequence.steps_with_calls_resolved]
        )

    @property
    def name(self) -> str:
        """Create a default name if none is present."""
        return self.custom_name or Flow.create_default_name(self.id)

    def is_startable(
        self,
        context: Optional[Dict[Text, Any]] = None,
        slots: Optional[Dict[Text, Slot]] = None,
    ) -> bool:
        """Return whether the start condition is satisfied.

        Args:
            context: The context data to evaluate the starting conditions against.
            slots: The slots to evaluate the starting conditions against.

        Returns:
            Whether the start condition is satisfied.
        """
        context = context or {}
        slots = slots or {}
        simplified_slots = {slot.name: slot.value for slot in slots.values()}

        # If no starting condition exists, the flow is always startable.
        if not self.guard_condition:
            return True

        # if a flow guard condition exists and the flow was started via a link,
        # e.g. is currently active, the flow is startable
        if context.get("flow_id") == self.id:
            return True

        try:
            predicate = Predicate(self.guard_condition)
            is_startable = predicate.evaluate(
                {"context": context, "slots": simplified_slots}
            )
            structlogger.debug(
                "command_generator.validate_flow_starting_conditions.result",
                predicate=predicate.description(),
                is_startable=is_startable,
                flow_id=self.id,
            )
            return is_startable
        # if there is any kind of exception when evaluating the predicate, the flow
        # is not startable
        except (TypeError, Exception) as e:
            structlogger.error(
                "command_generator.validate_flow_starting_conditions.error",
                predicate=self.guard_condition,
                context=context,
                slots=slots,
                error=str(e),
            )
            return False

    def has_action_step(self, action: Text) -> bool:
        """Check whether the flow has an action step with the given action."""
        for step in self.steps:
            if isinstance(step, ActionFlowStep) and step.action == action:
                return True
        return False

    def is_startable_only_via_link(self) -> bool:
        """Determines if the flow can be initiated exclusively through a link.

        This condition is met when a guard condition exists and is
        consistently evaluated to `False` (e.g. `if: False`).

        Returns:
            A boolean indicating if the flow initiation is link-based only.
        """
        if (
            self.guard_condition is None
            or self._contains_variables_in_guard_condition()
        ):
            return False

        try:
            predicate = Predicate(self.guard_condition)
            is_startable_via_link = not predicate.evaluate({})
            structlogger.debug(
                "flow.is_startable_only_via_link.result",
                predicate=self.guard_condition,
                is_startable_via_link=is_startable_via_link,
                flow_id=self.id,
            )
            return is_startable_via_link
        # if there is any kind of exception when evaluating the predicate, the flow
        # is not startable by link or by any other means.
        except (TypeError, Exception) as e:
            structlogger.error(
                "flow.is_startable_only_via_link.error",
                predicate=self.guard_condition,
                error=str(e),
                flow_id=self.id,
            )
            return False

    def _contains_variables_in_guard_condition(self) -> bool:
        """Determines if the guard condition contains dynamic literals.

        I.e. literals that cannot be statically resolved, indicating a variable.

        Returns:
            True if dynamic literals are present, False otherwise.
        """
        from pypred import ast
        from pypred.tiler import tile, SimplePattern

        if not self.guard_condition:
            return False

        predicate = Predicate(self.guard_condition)

        # find all literals in the AST tree
        literals = []
        tile(
            ast=predicate.ast,
            patterns=[SimplePattern("types:Literal")],
            func=lambda _, literal: literals.append(literal),
        )

        # check if there is a literal that cannot be statically resolved (variable)
        for literal in literals:
            if type(predicate.static_resolve(literal.value)) == ast.Undefined:
                return True

        return False
