from __future__ import annotations

import re
import typing
from collections import defaultdict
from typing import Optional, Set, Text, List

from rasa.shared.constants import (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX,
    RASA_PATTERN_HUMAN_HANDOFF,
)
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR,
)
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.flow_step import (
    FlowStep,
)
from rasa.shared.core.flows.flow_step_links import (
    BranchingFlowStepLink,
    IfFlowStepLink,
    ElseFlowStepLink,
)
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.flows.steps.constants import CONTINUE_STEP_PREFIX, DEFAULT_STEPS
from rasa.shared.core.flows.steps.link import LinkFlowStep
from rasa.shared.exceptions import RasaException

if typing.TYPE_CHECKING:
    from rasa.shared.core.flows.flows_list import FlowsList

FLOW_ID_REGEX = r"""^[a-zA-Z0-9_][a-zA-Z0-9_-]*?$"""


class UnreachableFlowStepException(RasaException):
    """Raised when a flow step is unreachable."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' can not be reached "
            f"from the start step. Please make sure that all steps can be reached "
            f"from the start step, e.g. by "
            f"checking that another step points to this step."
        )


class MissingNextLinkException(RasaException):
    """Raised when a flow step is missing a next link."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' is missing a `next`. "
            f"As a last step of a branch, it is required to have one. "
        )


class ReservedFlowStepIdException(RasaException):
    """Raised when a flow step is using a reserved id."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' is using the reserved id "
            f"'{self.step_id}'. Please use a different id for your step."
        )


class DuplicatedStepIdException(RasaException):
    """Raised when a flow step is using the same id as another step."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' is using the same id as "
            f"another step. Step ids must be unique across all steps of a flow. "
            f"Please use a different id for your step."
        )


class MissingElseBranchException(RasaException):
    """Raised when a flow step is missing an else branch."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' is missing an `else` "
            f"branch. If a steps `next` statement contains an `if` it always "
            f"also needs an `else` branch. Please add the missing `else` branch."
        )


class NoNextAllowedForLinkException(RasaException):
    """Raised when a flow step has a next link but is not allowed to have one."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Link step '{self.step_id}' in flow '{self.flow_id}' has a `next` but "
            f"as a link step is not allowed to have one."
        )


class ReferenceToPatternException(RasaException):
    """Raised when a flow step is referencing a pattern, which is not allowed."""

    def __init__(
        self, referenced_pattern: str, flow_id: str, step_id: str, call_step: bool
    ) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.referenced_pattern = referenced_pattern
        self.call_step = call_step

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        message = (
            f"Step '{self.step_id}' in flow '{self.flow_id}' is referencing a "
            f"pattern '{self.referenced_pattern}', which is not allowed. "
        )
        if self.call_step:
            return message + "Patterns can not be used as a target for a call step."
        else:
            return message + (
                "All patterns, except for 'pattern_human_handoff', can "
                "not be used as a target for a link step."
            )


class PatternReferencedPatternException(RasaException):
    """Raised when a pattern is referencing a pattern, which is not allowed."""

    def __init__(self, flow_id: str, step_id: str, call_step: bool) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.call_step = call_step

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        message = (
            f"Step '{self.step_id}' in pattern '{self.flow_id}' is referencing a "
            f"pattern which is not allowed. "
        )
        if self.call_step:
            return message + "Patterns can not use call steps to other patterns."
        else:
            return message + (
                "Patterns can not use link steps to other patterns. "
                "Exception: patterns can link to 'pattern_human_handoff'."
            )


class PatternReferencedFlowException(RasaException):
    """Raised when a pattern is referencing a flow, which is not allowed."""

    def __init__(self, flow_id: str, step_id: str, call_step: bool) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.call_step = call_step

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        message = (
            f"Step '{self.step_id}' in pattern '{self.flow_id}' is referencing a flow "
            f"which is not allowed. "
        )
        if self.call_step:
            return message + "Patterns can not use call steps."
        else:
            return message + (
                "'pattern_internal_error' can not use link steps to user flows."
            )


class NoLinkAllowedInCalledFlowException(RasaException):
    """Raised when a flow is called from another flow but is also using a link."""

    def __init__(self, step_id: str, flow_id: str, called_flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.called_flow_id = called_flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Flow '{self.flow_id}' is calling another flow (call step). "
            f"The flow that is getting called ('{self.called_flow_id}') is "
            f"using a link step, which is not allowed. "
            f"Either this flow can not be called or the link step in {self.step_id} "
            f"needs to be removed."
        )


class UnresolvedFlowException(RasaException):
    """Raised when a flow is called or linked from another flow but doesn't exist."""

    def __init__(self, flow_id: str, calling_flow_id: str, step_id: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id
        self.calling_flow_id = calling_flow_id
        self.step_id = step_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Flow '{self.flow_id}' is called or linked from flow "
            f"'{self.calling_flow_id}' in step '{self.step_id}', "
            f"but it doesn't exist. "
            f"Please make sure that a flow with id '{self.flow_id}' exists."
        )


class UnresolvedFlowStepIdException(RasaException):
    """Raised when a flow step is referenced, but its id can not be resolved."""

    def __init__(
        self, step_id: str, flow_id: str, referenced_from_step_id: Optional[str]
    ) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.referenced_from_step_id = referenced_from_step_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        if self.referenced_from_step_id:
            exception_message = (
                f"Step with id '{self.step_id}' could not be resolved. "
                f"'Step '{self.referenced_from_step_id}' in flow '{self.flow_id}' "
                f"referenced this step but it does not exist. "
            )
        else:
            exception_message = (
                f"Step '{self.step_id}' in flow '{self.flow_id}' can not be resolved. "
            )

        return exception_message + (
            "Please make sure that the step is defined in the same flow."
        )


class EmptyStepSequenceException(RasaException):
    """Raised when an empty step sequence is encountered."""

    def __init__(self, flow_id: str, step_id: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id
        self.step_id = step_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Encountered an empty step sequence in flow '{self.flow_id}' "
            f"and step '{self.step_id}'."
        )


class EmptyFlowException(RasaException):
    """Raised when a flow is completely empty."""

    def __init__(self, flow_id: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return f"Flow '{self.flow_id}' does not have any steps."


class DuplicateNLUTriggerException(RasaException):
    """Raised when multiple flows can be started by the same intent."""

    def __init__(self, intent: str, flow_names: List[str]) -> None:
        """Initializes the exception."""
        self.intent = intent
        self.flow_names = flow_names

    def __str__(self) -> Text:
        """Return a string representation of the exception."""
        return (
            f"The intent '{self.intent}' is used as 'nlu_trigger' "
            f"in multiple flows: {self.flow_names}. "
            f"An intent should just trigger one flow, not multiple."
        )


class SlotNamingException(RasaException):
    """Raised when a slot name to be collected does not adhere to naming convention."""

    def __init__(self, flow_id: str, step_id: str, slot_name: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id
        self.step_id = step_id
        self.slot_name = slot_name

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"For the flow '{self.flow_id}', collect step '{self.step_id}' "
            f"the slot name was set to : {self.slot_name}, while it has "
            f"to adhere to the following pattern: [a-zA-Z_][a-zA-Z0-9_-]*?."
        )


class FlowIdNamingException(RasaException):
    """Raised when a flow ID defined does not adhere to naming convention."""

    def __init__(self, flow_id: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"The flow ID was set to : {self.flow_id}, while it has "
            f"to adhere to the following pattern: [a-zA-Z0-9_][a-zA-Z0-9_-]*?."
        )


def validate_flow(flow: Flow) -> None:
    """Validates the flow configuration.

    This ensures that the flow semantically makes sense. E.g. it
    checks:
        - whether all next links point to existing steps
        - whether all steps can be reached from the start step
    """
    validate_flow_not_empty(flow)
    validate_no_empty_step_sequences(flow)
    validate_all_steps_next_property(flow)
    validate_all_next_ids_are_available_steps(flow)
    validate_all_steps_can_be_reached(flow)
    validate_all_branches_have_an_else(flow)
    validate_not_using_builtin_ids(flow)
    validate_slot_names_to_be_collected(flow)
    validate_flow_id(flow)


def validate_flow_not_empty(flow: Flow) -> None:
    """Validate that the flow is not empty."""
    if len(flow.steps) == 0:
        raise EmptyFlowException(flow.id)


def validate_no_empty_step_sequences(flow: Flow) -> None:
    """Validate that the flow does not have any empty step sequences."""
    for step in flow.steps:
        for link in step.next.links:
            if (
                isinstance(link, BranchingFlowStepLink)
                and isinstance(link.target_reference, FlowStepSequence)
                and len(link.target_reference.child_steps) == 0
            ):
                raise EmptyStepSequenceException(flow.id, step.id)


def validate_not_using_builtin_ids(flow: Flow) -> None:
    """Validates that the flow does not use any of the build in ids."""
    for step in flow.steps:
        if step.id in DEFAULT_STEPS or step.id.startswith(CONTINUE_STEP_PREFIX):
            raise ReservedFlowStepIdException(step.id, flow.id)


def validate_all_branches_have_an_else(flow: Flow) -> None:
    """Validates that all branches have an else link."""
    for step in flow.steps:
        links = step.next.links

        has_an_if = any(isinstance(link, IfFlowStepLink) for link in links)
        has_an_else = any(isinstance(link, ElseFlowStepLink) for link in links)

        if has_an_if and not has_an_else:
            raise MissingElseBranchException(step.id, flow.id)


def validate_all_steps_next_property(flow: Flow) -> None:
    """Validates that every step that must have a `next` has one."""
    for step in flow.steps:
        if isinstance(step, LinkFlowStep):
            # link steps can't have a next link!
            if not step.next.no_link_available():
                raise NoNextAllowedForLinkException(step.id, flow.id)
        elif step.next.no_link_available():
            # all other steps should have a next link
            raise MissingNextLinkException(step.id, flow.id)


def validate_all_next_ids_are_available_steps(flow: Flow) -> None:
    """Validates that all next links point to existing steps."""
    available_steps = {step.id for step in flow.steps} | DEFAULT_STEPS
    for step in flow.steps:
        for link in step.next.links:
            if link.target not in available_steps:
                raise UnresolvedFlowStepIdException(link.target, flow.id, step.id)


def validate_all_steps_can_be_reached(flow: Flow) -> None:
    """Validates that all steps can be reached from the start step."""

    def _reachable_steps(
        step: Optional[FlowStep], reached_steps: Set[Text]
    ) -> Set[Text]:
        """Validates that the given step can be reached from the start step."""
        if step is None or step.id in reached_steps:
            return reached_steps

        reached_steps.add(step.id)
        for link in step.next.links:
            reached_steps = _reachable_steps(
                flow.step_by_id(link.target), reached_steps
            )
        return reached_steps

    reached_steps = _reachable_steps(flow.first_step_in_flow(), set())

    for step in flow.steps:
        if step.id not in reached_steps:
            raise UnreachableFlowStepException(step.id, flow.id)


def validate_nlu_trigger(flows: List[Flow]) -> None:
    """Validates that an intent can just trigger one flow."""
    nlu_trigger_to_flows = defaultdict(list)

    for flow in flows:
        intents = flow.get_trigger_intents()
        for intent in intents:
            nlu_trigger_to_flows[intent].append(flow.name)

    for intent, flow_names in nlu_trigger_to_flows.items():
        if len(flow_names) > 1:
            raise DuplicateNLUTriggerException(intent, flow_names)


def validate_link_in_call_restriction(flows: "FlowsList") -> None:
    """Validates that a flow is not called from another flow and uses a link step."""

    def does_flow_use_link(flow_id: str) -> bool:
        if flow := flows.flow_by_id(flow_id):
            for step in flow.steps:
                if isinstance(step, LinkFlowStep):
                    return True
        return False

    for flow in flows.underlying_flows:
        for step in flow.steps:
            if isinstance(step, CallFlowStep) and does_flow_use_link(step.call):
                raise NoLinkAllowedInCalledFlowException(step.id, flow.id, step.call)


def validate_called_flows_exists(flows: "FlowsList") -> None:
    """Validates that all called flows exist."""
    for flow in flows.underlying_flows:
        for step in flow.steps:
            if not isinstance(step, CallFlowStep):
                continue

            if flows.flow_by_id(step.call) is None:
                raise UnresolvedFlowException(step.call, flow.id, step.id)


def validate_linked_flows_exists(flows: "FlowsList") -> None:
    """Validates that all linked flows exist."""
    for flow in flows.underlying_flows:
        for step in flow.steps:
            if not isinstance(step, LinkFlowStep):
                continue

            # It might be that the flows do not contain the default rasa patterns, but
            # only the user flows. Manually check for `pattern_human_handoff` as this
            # pattern can be linked to and it is part of the default patterns of rasa.
            if (
                flows.flow_by_id(step.link) is None
                and step.link != RASA_PATTERN_HUMAN_HANDOFF
            ):
                raise UnresolvedFlowException(step.link, flow.id, step.id)


def validate_patterns_are_not_called_or_linked(flows: "FlowsList") -> None:
    """Validates that patterns are never called or linked.

    Exception: pattern_human_handoff can be linked.
    """
    for flow in flows.underlying_flows:
        for step in flow.steps:
            if (
                isinstance(step, LinkFlowStep)
                and step.link.startswith(RASA_DEFAULT_FLOW_PATTERN_PREFIX)
                and step.link != RASA_PATTERN_HUMAN_HANDOFF
            ):
                raise ReferenceToPatternException(
                    step.link, flow.id, step.id, call_step=False
                )

            if isinstance(step, CallFlowStep) and step.call.startswith(
                RASA_DEFAULT_FLOW_PATTERN_PREFIX
            ):
                raise ReferenceToPatternException(
                    step.call, flow.id, step.id, call_step=True
                )


def validate_patterns_are_not_calling_or_linking_other_flows(
    flows: "FlowsList",
) -> None:
    """Validates that patterns do not contain call or link steps.

    Link steps to user flows are allowed for all patterns but 'pattern_internal_error'.
    Link steps to other patterns, except for 'pattern_human_handoff', are forbidden.
    """
    for flow in flows.underlying_flows:
        if not flow.is_rasa_default_flow:
            continue
        for step in flow.steps:
            if isinstance(step, LinkFlowStep):
                if step.link == RASA_PATTERN_HUMAN_HANDOFF:
                    # links to 'pattern_human_handoff' are allowed
                    continue
                if step.link.startswith(RASA_DEFAULT_FLOW_PATTERN_PREFIX):
                    # all other patterns are allowed to link to user flows, but not
                    # to other patterns
                    raise PatternReferencedPatternException(
                        flow.id, step.id, call_step=False
                    )
                if flow.id == RASA_PATTERN_INTERNAL_ERROR:
                    # 'pattern_internal_error' is not allowed to link at all
                    raise PatternReferencedFlowException(
                        flow.id, step.id, call_step=False
                    )
            if isinstance(step, CallFlowStep):
                if step.call.startswith(RASA_DEFAULT_FLOW_PATTERN_PREFIX):
                    raise PatternReferencedPatternException(
                        flow.id, step.id, call_step=True
                    )
                else:
                    raise PatternReferencedFlowException(
                        flow.id, step.id, call_step=True
                    )


def validate_step_ids_are_unique(flows: "FlowsList") -> None:
    """Validates that step ids are unique within a flow and any called flows."""
    for flow in flows.underlying_flows:
        used_ids: Set[str] = set()

        # check that the ids used in the flow are unique
        for step in flow.steps:
            if step.id in used_ids:
                raise DuplicatedStepIdException(step.id, flow.id)

            used_ids.add(step.id)


def validate_slot_names_to_be_collected(flow: Flow) -> None:
    """Validates that slot names to be collected comply with a specified regex."""
    slot_re = re.compile(r"""^[a-zA-Z_][a-zA-Z0-9_-]*?$""")
    for step in flow.steps:
        if isinstance(step, CollectInformationFlowStep):
            slot_name = step.collect
            if not slot_re.search(slot_name):
                raise SlotNamingException(flow.id, step.id, slot_name)


def validate_flow_id(flow: Flow) -> None:
    """Validates if the flow id comply with a specified regex.
    Flow IDs can start with an alphanumeric character or an underscore.
    Followed by zero or more alphanumeric characters, hyphens, or underscores.

    Args:
        flow: The flow to validate.

    Raises:
        FlowIdNamingException: If the flow id does not comply with the regex.
    """
    flow_re = re.compile(FLOW_ID_REGEX)
    if not flow_re.search(flow.id):
        raise FlowIdNamingException(flow.id)
