from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Text, Optional, Dict, Any, List, Set, Union

import structlog
from pypred import Predicate

import rasa.shared.utils.io
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.flows.flow_path import PathNode, FlowPath, FlowPathsList
from rasa.shared.core.flows.flow_step import FlowStep
from rasa.shared.core.flows.flow_step_links import (
    FlowStepLink,
    StaticFlowStepLink,
    IfFlowStepLink,
    ElseFlowStepLink,
)
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.nlu_trigger import NLUTriggers
from rasa.shared.core.flows.steps import (
    CollectInformationFlowStep,
    EndFlowStep,
    StartFlowStep,
    ActionFlowStep,
)
from rasa.shared.core.flows.steps.constants import (
    CONTINUE_STEP_PREFIX,
    START_STEP,
    END_STEP,
)
from rasa.shared.core.flows.steps.continuation import ContinueFlowStep
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
    file_path: Optional[str] = None
    """The path to the file where the flow is stored."""

    @staticmethod
    def from_json(
        flow_id: Text,
        data: Dict[Text, Any],
        file_path: Optional[Union[str, Path]] = None,
    ) -> Flow:
        """Create a Flow object from serialized data.

        Args:
            flow_id: id of the flow
            data: data for a Flow object in a serialized format.
            file_path: the file path of the flow

        Returns:
            A Flow object.
        """
        step_sequence = FlowStepSequence.from_json(data.get("steps"))
        nlu_triggers = NLUTriggers.from_json(data.get("nlu_trigger"))

        if file_path and isinstance(file_path, Path):
            file_path = str(file_path)

        return Flow(
            id=flow_id,
            custom_name=data.get("name"),
            description=data.get("description"),
            always_include_in_prompt=data.get("always_include_in_prompt"),
            # str or bool are permitted in the flow schema but internally we want a str
            guard_condition=str(data["if"]) if "if" in data else None,
            step_sequence=Flow.resolve_default_ids(step_sequence),
            nlu_triggers=nlu_triggers,
            # If we are reading the flows in after training the file_path is part of
            # data. When the model is trained, take the provided file_path.
            file_path=data.get("file_path") if "file_path" in data else file_path,
        )

    def get_full_name(self) -> str:
        if self.file_path:
            return f"{self.file_path}::{self.name}"
        return self.name

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
        if self.file_path:
            data["file_path"] = self.file_path

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
    def custom_actions(self) -> Set[str]:
        """Retrieve all custom actions of this flow."""
        return {
            step.custom_action
            for step in self.step_sequence.steps_with_calls_resolved
            if isinstance(step, ActionFlowStep) and step.custom_action is not None
        }

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

    def extract_all_paths(self) -> FlowPathsList:
        """Extracts all possible flow paths.

        Extracts all possible flow paths from a given flow structure by
        recursively exploring each step.
        This function initializes an empty list to collect paths, an empty path list,
        and a set of visited step IDs to prevent revisiting steps.
        It calls `go_over_steps` to recursively explore and fill the paths list.
        """
        flow_paths_list = FlowPathsList(self.id, paths=[])
        steps: List[FlowStep] = self.steps
        current_path: FlowPath = FlowPath(flow=self.id, nodes=[])
        step_ids_visited: Set[str] = set()

        self._go_over_steps(steps, current_path, flow_paths_list, step_ids_visited)

        if not flow_paths_list.is_path_part_of_list(current_path):
            flow_paths_list.paths.append(copy.deepcopy(current_path))

        structlogger.debug(
            "shared.core.flows.flow.extract_all_paths",
            comment="Extraction complete",
            number_of_paths=len(flow_paths_list.paths),
            flow_name=self.name,
        )
        return flow_paths_list

    def _go_over_steps(
        self,
        steps_to_go: Union[str, List[FlowStep]],
        current_path: FlowPath,
        completed_paths: FlowPathsList,
        step_ids_visited: Set[str],
    ) -> None:
        """Processes the flow steps recursively.

        Either following direct step IDs or handling conditions, and adds complete
        paths to the collected_paths.

        Args:
            steps_to_go: Either a direct step ID or a list of steps to process.
            current_path: The current path being constructed.
            completed_paths: The list where completed paths are added.
            step_ids_visited: A set of step IDs that have been visited to avoid cycles.

        Returns:
            None: This function modifies collected_paths in place by appending new paths
            as they are found.
        """
        # Case 1: If the steps_to_go is a custom_id string
        # This happens when a "next" of, for example, a IfFlowStepLink is targeting
        # a specific step by id
        if isinstance(steps_to_go, str):
            for i, step in enumerate(self.steps):
                # We don't need to check for 'id' as a link can only happen to a
                # custom id.
                if step.custom_id == steps_to_go:
                    self._go_over_steps(
                        self.steps[i:], current_path, completed_paths, step_ids_visited
                    )

        # Case 2: If steps_to_go is a list of steps
        else:
            for i, step in enumerate(steps_to_go):
                # 1. Check if the step is relevant for testable_paths extraction.
                # We only create new path nodes for ActionFlowStep and
                # CollectInformationFlowStep because these are externally visible
                # changes in the assistant's behaviour (trackable in the e2e tests).
                # For other flow steps, we only follow their links.
                # We decided to ignore calls to other flows in our coverage analysis.
                if not isinstance(step, (CollectInformationFlowStep, ActionFlowStep)):
                    self._handle_links(
                        step.next.links,
                        current_path,
                        completed_paths,
                        step_ids_visited,
                    )
                    continue

                # 2. Check if already visited this custom step id
                # in order to keep track of loops
                if step.custom_id is not None and step.custom_id in step_ids_visited:
                    if not completed_paths.is_path_part_of_list(current_path):
                        completed_paths.paths.append(copy.deepcopy(current_path))
                    return  # Stop traversing this path if we've revisited a step
                elif step.custom_id is not None:
                    step_ids_visited.add(step.custom_id)

                # 3. Append step info to the path
                current_path.nodes.append(
                    PathNode(
                        flow=current_path.flow,
                        step_id=step.id,
                        lines=step.metadata["line_numbers"],
                    )
                )

                # 4. Check if 'END' branch
                if (
                    len(step.next.links) == 1
                    and isinstance(step.next.links[0], StaticFlowStepLink)
                    and step.next.links[0].target == END_STEP
                ):
                    if not completed_paths.is_path_part_of_list(current_path):
                        completed_paths.paths.append(copy.deepcopy(current_path))
                    return
                else:
                    self._handle_links(
                        step.next.links,
                        current_path,
                        completed_paths,
                        step_ids_visited,
                    )

    def _handle_links(
        self,
        links: List[FlowStepLink],
        path: FlowPath,
        collected_paths: FlowPathsList,
        step_ids_visited: set,
    ) -> None:
        """Processes the next step in a flow.

        Potentially recursively calling itself to handle conditional paths and
        branching.

        Args:
            links: Links listed in the "next" attribute.
            path: The current path taken in the flow.
            collected_paths: A list of paths collected so far.
            step_ids_visited: A set of step IDs that have already been visited
                to avoid loops.

        Returns:
            None: Modifies collected_paths in place by appending new paths
            as they are completed.
        """
        steps = self.steps

        for link in links:
            # Direct step id reference
            if isinstance(link, StaticFlowStepLink):
                # Find this id in the flow steps and restart from there
                for i, step in enumerate(steps):
                    if step.id == link.target_step_id:
                        self._go_over_steps(
                            steps[i:],
                            copy.deepcopy(path),
                            collected_paths,
                            copy.deepcopy(step_ids_visited),
                        )

            # If conditions
            elif isinstance(link, (IfFlowStepLink, ElseFlowStepLink)):
                # Handling conditional paths
                target_steps: Union[str, List[FlowStep]]
                if isinstance(link.target_reference, FlowStepSequence):
                    target_steps = link.target_reference.child_steps
                else:
                    target_steps = link.target_reference

                self._go_over_steps(
                    target_steps,
                    copy.deepcopy(path),
                    collected_paths,
                    copy.deepcopy(step_ids_visited),
                )
