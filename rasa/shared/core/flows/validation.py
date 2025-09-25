from __future__ import annotations

import re
import typing
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List, Optional, Set, Text, Tuple

from jinja2 import Template

from rasa.shared.constants import (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX,
    RASA_PATTERN_CHITCHAT,
    RASA_PATTERN_HUMAN_HANDOFF,
    RASA_PATTERN_INTERNAL_ERROR,
)
from rasa.shared.core.flows.constants import (
    KEY_MAPPING_INPUT,
    KEY_MAPPING_OUTPUT,
    KEY_MAPPING_SLOT,
)
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.flow_step import (
    FlowStep,
)
from rasa.shared.core.flows.flow_step_links import (
    BranchingFlowStepLink,
    ElseFlowStepLink,
    IfFlowStepLink,
)
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.flows.steps.constants import CONTINUE_STEP_PREFIX, DEFAULT_STEPS
from rasa.shared.core.flows.steps.link import LinkFlowStep
from rasa.shared.core.flows.steps.set_slots import SetSlotsFlowStep
from rasa.shared.core.flows.utils import (
    get_duplicate_slot_persistence_config_error_message,
    get_invalid_slot_persistence_config_error_message,
    warn_deprecated_collect_step_config,
)
from rasa.shared.exceptions import RasaException

if typing.TYPE_CHECKING:
    from rasa.shared.core.domain import Domain
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


class InvalidExitIfConditionException(RasaException):
    """Raised when an exit_if condition is invalid."""

    def __init__(self, step_id: str, flow_id: str, condition: str, reason: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.condition = condition
        self.reason = reason

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Invalid exit_if condition '{self.condition}' in step '{self.step_id}' "
            f"of flow '{self.flow_id}': {self.reason}. "
            f"Please ensure that exit_if conditions contain at least one slot "
            f"reference and use only defined slots with valid predicates."
        )


class ExitIfExclusivityException(RasaException):
    """Raised when a call step with exit_if has other properties."""

    def __init__(
        self, step_id: str, flow_id: str, conflicting_properties: List[str]
    ) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.conflicting_properties = conflicting_properties

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        conflicting_properties_str = ", ".join(self.conflicting_properties)
        return (
            f"Call step '{self.step_id}' in flow '{self.flow_id}' has an 'exit_if' "
            f"property but also has other properties: {conflicting_properties_str}. "
            f"A call step with 'exit_if' cannot have any other properties."
        )


class DuplicatedFlowIdException(RasaException):
    """Raised when a flow is using the same id as another flow."""

    def __init__(
        self, flow_id: str, first_file_path: str, second_file_path: str
    ) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id
        self.first_file_path = first_file_path
        self.second_file_path = second_file_path

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        if self.first_file_path == self.second_file_path:
            return (
                f"Flow '{self.flow_id}' is used twice in `{self.first_file_path}`. "
                f"Please make sure flow IDs are unique across all files."
            )
        return (
            f"Flow '{self.flow_id}' is used in both "
            f"`{self.first_file_path}` and `{self.second_file_path}`. "
            f"Please make sure flow IDs are unique across all files."
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
                "Patterns cannot be used as a target in link steps, except for "
                "'pattern_human_handoff', which may be linked from both user-defined "
                "flows and other patterns. 'pattern_chitchat' may only be linked "
                "from other patterns."
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


class UnresolvedLinkFlowException(RasaException):
    """Raised when a flow is called or linked from another flow but doesn't exist."""

    def __init__(self, flow_id: str, calling_flow_id: str, step_id: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id
        self.calling_flow_id = calling_flow_id
        self.step_id = step_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Flow '{self.flow_id}' is linked from flow "
            f"'{self.calling_flow_id}' in step '{self.step_id}', "
            f"but it doesn't exist. "
            f"Please make sure that a flow with id '{self.flow_id}' exists."
        )


class UnresolvedCallStepException(RasaException):
    """Raised when a call step doesn't have a reference to an existing flow or agent."""

    def __init__(
        self, call_step_argument: str, calling_flow_id: str, step_id: str
    ) -> None:
        """Initializes the exception."""
        self.call_step_argument = call_step_argument
        self.calling_flow_id = calling_flow_id
        self.step_id = step_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"The call step '{self.step_id}' in flow '{self.calling_flow_id}' "
            f"is invalid: there is no flow or agent with the "
            f"id '{self.call_step_argument}'. "
            f"Please make sure that the call step argument is a valid flow id "
            f"or an agent id."
        )


class InvalidMCPServerReferenceException(RasaException):
    """Raised when a call step references a non-existent MCP server."""

    def __init__(
        self, mcp_server_name: str, calling_flow_id: str, step_id: str
    ) -> None:
        """Initializes the exception."""
        self.mcp_server_name = mcp_server_name
        self.calling_flow_id = calling_flow_id
        self.step_id = step_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Call step '{self.step_id}' in flow '{self.calling_flow_id}' "
            f"references MCP server '{self.mcp_server_name}' which does not exist "
            f"in endpoints.yml. Please make sure that the MCP server is properly "
            f"configured in endpoints.yml."
        )


class InvalidMCPMappingSlotException(RasaException):
    """Raised when MCP tool mapping references non-existent slots."""

    def __init__(
        self,
        step_id: str,
        flow_id: str,
        invalid_input_slots: Set[str],
        invalid_output_slots: Set[str],
    ) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.invalid_input_slots = invalid_input_slots
        self.invalid_output_slots = invalid_output_slots

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        error_parts = []

        if self.invalid_input_slots:
            input_slots_str = ", ".join(sorted(self.invalid_input_slots))
            error_parts.append(f"input slots: {input_slots_str}")

        if self.invalid_output_slots:
            output_slots_str = ", ".join(sorted(self.invalid_output_slots))
            error_parts.append(f"output slots: {output_slots_str}")

        slots_info = " ".join(error_parts)
        return (
            f"Call step '{self.step_id}' in flow '{self.flow_id}' has slots not "
            f"defined in the domain: {slots_info}"
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


class DuplicateSlotPersistConfigException(RasaException):
    """Raised when a slot persist configuration is duplicated."""

    def __init__(self, flow_id: str, collect_step: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id
        self.collect_step = collect_step

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return get_duplicate_slot_persistence_config_error_message(
            self.flow_id, self.collect_step
        )


class InvalidPersistSlotsException(RasaException):
    """Raised when a slot persist configuration is duplicated."""

    def __init__(self, flow_id: str, invalid_slots: Set[str]) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id
        self.invalid_slots = invalid_slots

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return get_invalid_slot_persistence_config_error_message(
            self.flow_id, self.invalid_slots
        )


@dataclass
class ValidationResult:
    is_valid: bool
    invalid_slots: Set[str]


def validate_flow(flow: Flow) -> None:
    """Validates the flow configuration.

    This ensures that the flow semantically makes sense. E.g. it
    checks:
        - whether all next links point to existing steps
        - whether all steps can be reached from the start step
    """
    from rasa.cli.utils import is_skip_validation_flag_set

    validate_flow_not_empty(flow)
    validate_no_empty_step_sequences(flow)
    validate_all_steps_next_property(flow)
    validate_all_next_ids_are_available_steps(flow)
    validate_all_steps_can_be_reached(flow)
    validate_all_branches_have_an_else(flow)
    validate_not_using_builtin_ids(flow)
    validate_slot_names_to_be_collected(flow)
    validate_flow_id(flow)

    if is_skip_validation_flag_set():
        # we only want to run this validation if the --skip-validation flag is used
        # during training because Flow Validation exceptions are raised one by one
        # as opposed to all at once with the Validator class
        validate_slot_persistence_configuration(flow)


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


def validate_call_steps(flows: "FlowsList") -> None:
    """Validates that all called flows/agents exist and properties are valid."""
    for flow in flows.underlying_flows:
        for step in flow.steps:
            if not isinstance(step, CallFlowStep):
                continue

            _is_step_calling_agent = step.is_calling_agent()
            _is_step_calling_mcp_tool = step.has_mcp_tool_properties()

            if (
                not _is_step_calling_agent
                and flows.flow_by_id(step.call) is None
                and not _is_step_calling_mcp_tool
            ):
                raise UnresolvedCallStepException(step.call, flow.id, step.id)

            if step.exit_if and not _is_step_calling_agent:
                # exit_if is only allowed for call steps that call an agent
                raise RasaException(
                    f"Call step '{step.id}' in flow '{flow.id}' has an 'exit_if' "
                    f"condition, but it is not calling an agent. "
                    f"'exit_if' is only allowed for call steps that call an agent."
                )


def validate_mcp_server_references(flows: "FlowsList") -> None:
    """Validates that MCP server references in call steps are valid."""
    for flow in flows.underlying_flows:
        for step in flow.steps:
            if not isinstance(step, CallFlowStep):
                continue

            # Only validate call steps that are trying to call MCP tools
            if step.has_mcp_tool_properties():
                # Check if the referenced MCP server exists
                from rasa.shared.utils.mcp.utils import mcp_server_exists

                if not mcp_server_exists(step.mcp_server):
                    raise InvalidMCPServerReferenceException(
                        step.mcp_server, flow.id, step.id
                    )


def validate_mcp_mapping_slots(
    flows: "FlowsList", domain: Optional["Domain"] = None
) -> None:
    """Validates that slots referenced in MCP tool mapping are defined in the domain.

    This function validates that all slots referenced in the input and output
    mapping of MCP tool calls are defined in the domain.

    Args:
        flows: The flows to validate.
        domain: The domain with slot definitions. If None, slot validation is skipped.

    Raises:
        InvalidMCPMappingSlotException: If any MCP mapping references undefined slots.
    """
    if domain is None:
        return

    domain_slots = {slot.name for slot in domain.slots}

    for flow in flows.underlying_flows:
        for step in flow.steps:
            if not isinstance(step, CallFlowStep):
                continue

            # Only validate call steps that are calling MCP tools
            if not step.has_mcp_tool_properties() or step.mapping is None:
                continue

            # Extract slot names from input mapping
            input_slots = set()
            if KEY_MAPPING_INPUT in step.mapping and isinstance(
                step.mapping[KEY_MAPPING_INPUT], list
            ):
                for input_item in step.mapping[KEY_MAPPING_INPUT]:
                    input_slots.add(input_item[KEY_MAPPING_SLOT])

            # Extract slot names from output mapping
            output_slots = set()
            if KEY_MAPPING_OUTPUT in step.mapping and isinstance(
                step.mapping[KEY_MAPPING_OUTPUT], list
            ):
                for output_item in step.mapping[KEY_MAPPING_OUTPUT]:
                    output_slots.add(output_item[KEY_MAPPING_SLOT])

            # Check for invalid slots in both input and output
            invalid_input_slots = input_slots - domain_slots
            invalid_output_slots = output_slots - domain_slots

            # Raise exception if any invalid slots are found
            if invalid_input_slots or invalid_output_slots:
                raise InvalidMCPMappingSlotException(
                    step.id, flow.id, invalid_input_slots, invalid_output_slots
                )


def _get_call_steps_with_exit_if(
    flows: "FlowsList",
) -> Iterator[Tuple["CallFlowStep", "Flow"]]:
    """Helper function to get all call steps with exit_if properties.

    Args:
        flows: The flows to search through.

    Yields:
        Tuples of (call_step, flow) for steps that have exit_if properties.
    """
    for flow in flows.underlying_flows:
        for step in flow.steps:
            if isinstance(step, CallFlowStep) and step.exit_if:
                yield step, flow


def validate_exit_if_conditions(
    flows: "FlowsList", domain: Optional["Domain"] = None
) -> None:
    """Validates that exit_if conditions are valid.

    This function validates:
    - Each condition contains at least one slot reference
    - Only defined slots are used within the conditions
    - The predicates are valid

    Args:
        flows: The flows to validate.
        domain: The domain with slot definitions. If None, slot validation is skipped.

    Raises:
        InvalidExitIfConditionException: If any exit_if condition is invalid.
    """
    for step, flow in _get_call_steps_with_exit_if(flows):
        for condition in step.exit_if:  # type: ignore[union-attr]
            if not isinstance(condition, str):
                raise InvalidExitIfConditionException(
                    step.id, flow.id, str(condition), "Condition must be a string"
                )

            # Check if condition contains at least one slot reference
            slot_references_regex = re.compile(r"\bslots\.\w+")
            slot_references = slot_references_regex.findall(condition)
            if not slot_references:
                raise InvalidExitIfConditionException(
                    step.id,
                    flow.id,
                    condition,
                    "Condition must contain at least one slot reference "
                    "(e.g., 'slots.slot_name')",
                )

            # Validate predicate syntax using pypred (always, regardless of domain)
            _validate_predicate_syntax_with_pypred(step.id, flow.id, condition)

            # Validate slot names if domain is provided
            if domain:
                domain_slots = {slot.name: slot for slot in domain.slots}
                for slot_ref in slot_references:
                    slot_name = slot_ref.split(".")[1]
                    if slot_name not in domain_slots:
                        raise InvalidExitIfConditionException(
                            step.id,
                            flow.id,
                            condition,
                            f"Slot '{slot_name}' is not defined in the domain",
                        )


def validate_exit_if_exclusivity(flows: "FlowsList") -> None:
    """Validates that call steps with exit_if don't have other properties.

    This function validates that call steps with an exit_if property cannot have
    any other properties besides the required 'call' property and standard flow
    step properties (id, next, metadata, description).

    Args:
        flows: The flows to validate.

    Raises:
        ExitIfExclusivityException: If a call step with exit_if has other properties.
    """
    for step, flow in _get_call_steps_with_exit_if(flows):
        # Check for conflicting properties
        conflicting_properties = []

        # Check for MCP-related properties
        if step.mcp_server is not None:
            conflicting_properties.append("mcp_server")
        if step.mapping is not None:
            conflicting_properties.append("mapping")

        if conflicting_properties:
            raise ExitIfExclusivityException(step.id, flow.id, conflicting_properties)


def validate_linked_flows_exists(flows: "FlowsList") -> None:
    """Validates that all linked flows exist."""
    for flow in flows.underlying_flows:
        for step in flow.steps:
            if not isinstance(step, LinkFlowStep):
                continue

            # It might be that the flows do not contain the default rasa patterns, but
            # only the user flows. Manually check for `pattern_human_handoff` and
            # 'pattern_chitchat' as these patterns can be linked to and are part of the
            # default patterns of rasa.
            if (
                flows.flow_by_id(step.link) is None
                # Allow linking to human-handoff from both patterns
                # and user-defined flows
                and step.link != RASA_PATTERN_HUMAN_HANDOFF
                # Allow linking to 'pattern_chitchat' only from other patterns
                and not (
                    flow.is_rasa_default_flow and step.link == RASA_PATTERN_CHITCHAT
                )
            ):
                raise UnresolvedLinkFlowException(step.link, flow.id, step.id)


def validate_patterns_are_not_called_or_linked(flows: "FlowsList") -> None:
    """Validates that patterns are never called or linked.

    Exception: pattern_human_handoff can be linked.
    """
    for flow in flows.underlying_flows:
        for step in flow.steps:
            if (
                isinstance(step, LinkFlowStep)
                and step.link.startswith(RASA_DEFAULT_FLOW_PATTERN_PREFIX)
                # Allow linking to human-handoff from both patterns
                # and user-defined flows
                and step.link != RASA_PATTERN_HUMAN_HANDOFF
                # Allow linking to 'pattern_chitchat' only from other patterns
                and not (
                    flow.is_rasa_default_flow and step.link == RASA_PATTERN_CHITCHAT
                )
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
    Link steps to other patterns, except for 'pattern_human_handoff' and
    'pattern_chitchat' are forbidden.
    """
    for flow in flows.underlying_flows:
        if not flow.is_rasa_default_flow:
            continue
        for step in flow.steps:
            if isinstance(step, LinkFlowStep):
                if step.link == RASA_PATTERN_HUMAN_HANDOFF:
                    # links to 'pattern_human_handoff' are allowed
                    continue
                if step.link == RASA_PATTERN_CHITCHAT:
                    # links to 'pattern_chitchat' are allowed
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


def validate_slot_persistence_configuration(flow: Flow) -> None:
    """Validates that slot persistence configuration is valid.

    Only slots used in either a collect step or a set_slot step can be persisted
    and the configuration can either be set at the flow level or the collect step level,
    but not both.

    Args:
        flow: The flow to validate.

    Raises:
        DuplicateSlotPersistConfigException: If slot persist config is duplicated.
    """

    def _is_persist_slots_valid(
        persist_slots: List[str], flow_slots: Set[str]
    ) -> ValidationResult:
        """Validates that the slots that should be persisted are used in the flow."""
        invalid_slots = set(persist_slots) - flow_slots
        is_valid = False if invalid_slots else True

        return ValidationResult(is_valid, invalid_slots)

    flow_id = flow.id
    persist_slots = flow.persisted_slots
    has_flow_level_persistence = True if persist_slots else False
    flow_slots = set()

    for step in flow.steps_with_calls_resolved:
        if isinstance(step, SetSlotsFlowStep):
            flow_slots.update([slot["key"] for slot in step.slots])
        elif isinstance(step, CollectInformationFlowStep):
            flow_slots.add(step.collect)
            if not step.reset_after_flow_ends:
                collect_step = step.collect
                warn_deprecated_collect_step_config()
                if has_flow_level_persistence:
                    raise DuplicateSlotPersistConfigException(flow_id, collect_step)

    if has_flow_level_persistence:
        result = _is_persist_slots_valid(persist_slots, flow_slots)
        if not result.is_valid:
            raise InvalidPersistSlotsException(flow_id, result.invalid_slots)


def _validate_predicate_syntax_with_pypred(
    step_id: str, flow_id: str, condition: str
) -> None:
    """Validates predicate syntax using pypred.

    This function validates that the exit_if condition has valid predicate syntax.
    Pypred catches syntax errors like double operators, invalid expressions, etc.

    Args:
        step_id: The ID of the step containing the condition.
        flow_id: The ID of the flow containing the step.
        condition: The exit_if condition string.

    Raises:
        InvalidExitIfConditionException: If pypred detects syntax errors.
    """
    try:
        from rasa.utils.pypred import Predicate

        # Create a simple test context for syntax validation.
        # This context works with all slot types since it's only used for:
        # 1. Template rendering: Basic structure for Jinja2 template rendering
        # 2. Syntax validation: Pypred validates predicate syntax
        #    without actual slot values
        test_context = {"slots": {"test_slot": "test_value"}}
        rendered_template = Template(condition).render(test_context)

        # Let pypred validate the predicate syntax
        predicate = Predicate(rendered_template)
        if not predicate.is_valid():
            raise InvalidExitIfConditionException(
                step_id,
                flow_id,
                condition,
                "Invalid predicate syntax",
            )

    except ImportError:
        # pypred not available, skip validation
        pass
    except Exception as e:
        # Re-raise pypred errors as predicate syntax errors
        raise InvalidExitIfConditionException(
            step_id,
            flow_id,
            condition,
            f"Invalid predicate syntax: {e}",
        )
