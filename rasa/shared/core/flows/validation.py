from typing import Optional, Set, Text

from rasa.shared.core.flows.exceptions import (
    EmptyFlowException,
    EmptyStepSequenceException,
    ReservedFlowStepIdException,
    MissingElseBranchException,
    NoNextAllowedForLinkException,
    MissingNextLinkException,
    UnresolvedFlowStepIdException,
    UnreachableFlowStepException,
)
from rasa.shared.core.flows.flow import (
    Flow,
    BranchBasedLink,
    DEFAULT_STEPS,
    CONTINUE_STEP_PREFIX,
    IfFlowLink,
    ElseFlowLink,
    LinkFlowStep,
    FlowStep,
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
    validate_not_using_buildin_ids(flow)


def validate_flow_not_empty(flow: Flow) -> None:
    """Validate that the flow is not empty."""
    if len(flow.steps) == 0:
        raise EmptyFlowException(flow.id)


def validate_no_empty_step_sequences(flow: Flow) -> None:
    """Validate that the flow does not have any empty step sequences."""
    for step in flow.steps:
        for link in step.next.links:
            if isinstance(link, BranchBasedLink) and link.target is None:
                raise EmptyStepSequenceException(flow.id, step.id)


def validate_not_using_buildin_ids(flow: Flow) -> None:
    """Validates that the flow does not use any of the build in ids."""
    for step in flow.steps:
        if step.id in DEFAULT_STEPS or step.id.startswith(CONTINUE_STEP_PREFIX):
            raise ReservedFlowStepIdException(step.id, flow.id)


def validate_all_branches_have_an_else(flow: Flow) -> None:
    """Validates that all branches have an else link."""
    for step in flow.steps:
        links = step.next.links

        has_an_if = any(isinstance(link, IfFlowLink) for link in links)
        has_an_else = any(isinstance(link, ElseFlowLink) for link in links)

        if has_an_if and not has_an_else:
            raise MissingElseBranchException(step.id, flow.id)


def validate_all_steps_next_property(flow: Flow) -> None:
    """Validates that every step has a next link."""
    for step in flow.steps:
        if isinstance(step, LinkFlowStep):
            # link steps can't have a next link!
            if not step.next.no_link_available():
                raise NoNextAllowedForLinkException(step.id, flow.id)
        elif step.next.no_link_available():
            # all other steps should have a next link
            raise MissingNextLinkException(step.id, flow.id)


def validate_all_next_ids_are_available_steps(flow) -> None:
    """Validates that all next links point to existing steps."""
    available_steps = {step.id for step in flow.steps} | DEFAULT_STEPS
    for step in flow.steps:
        for link in step.next.links:
            if link.target not in available_steps:
                raise UnresolvedFlowStepIdException(link.target, flow.id, step.id)


def validate_all_steps_can_be_reached(flow) -> None:
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
