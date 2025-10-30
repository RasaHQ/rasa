from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union

import structlog
from pydantic import BaseModel

import rasa.shared.utils.io
from rasa.engine.language import Language
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.flows.constants import (
    KEY_ALWAYS_INCLUDE_IN_PROMPT,
    KEY_CALLED_FLOW,
    KEY_DESCRIPTION,
    KEY_FILE_PATH,
    KEY_ID,
    KEY_IF,
    KEY_LINKED_FLOW,
    KEY_NAME,
    KEY_NLU_TRIGGER,
    KEY_PERSISTED_SLOTS,
    KEY_RUN_PATTERN_COMPLETED,
    KEY_STEPS,
    KEY_TRANSLATION,
)
from rasa.shared.core.flows.flow_path import FlowPath, FlowPathsList, PathNode
from rasa.shared.core.flows.flow_step import FlowStep
from rasa.shared.core.flows.flow_step_links import (
    ElseFlowStepLink,
    FlowStepLink,
    IfFlowStepLink,
    StaticFlowStepLink,
)
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.nlu_trigger import NLUTriggers
from rasa.shared.core.flows.steps import (
    ActionFlowStep,
    CallFlowStep,
    CollectInformationFlowStep,
    EndFlowStep,
    LinkFlowStep,
    StartFlowStep,
)
from rasa.shared.core.flows.steps.constants import (
    CONTINUE_STEP_PREFIX,
    END_STEP,
    START_STEP,
)
from rasa.shared.core.flows.steps.continuation import ContinueFlowStep
from rasa.shared.core.slots import Slot
from rasa.utils.pypred import Predicate

structlogger = structlog.get_logger()

DEFAULT_RUN_PATTERN_COMPLETED = True


class FlowLanguageTranslation(BaseModel):
    """Represents the translation of the flow properties in a specific language."""

    name: str
    """The human-readable name of the flow."""

    class Config:
        """Configuration for the FlowLanguageTranslation model."""

        extra = "ignore"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FlowLanguageTranslation):
            return self.name == other.name
        return False


@dataclass
class Flow:
    """Represents the configuration of a flow."""

    id: Text
    """The id of the flow."""
    custom_name: Optional[Text] = None
    """The human-readable name of the flow."""
    description: Optional[Text] = None
    """The description of the flow."""
    translation: Dict[Text, FlowLanguageTranslation] = field(default_factory=dict)
    """The translation of the flow properties in different languages."""
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
    persisted_slots: List[str] = field(default_factory=list)
    """The list of slots that should be persisted after the flow ends."""
    run_pattern_completed: bool = DEFAULT_RUN_PATTERN_COMPLETED
    """Whether the pattern_completed flow should be run after the flow ends."""
    metadata: Dict[Text, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Flow):
            return (
                self.id == other.id
                and self.custom_name == other.custom_name
                and self.description == other.description
                and self.translation == other.translation
                and self.guard_condition == other.guard_condition
                and self.step_sequence == other.step_sequence
                and self.nlu_triggers == other.nlu_triggers
                and self.always_include_in_prompt == other.always_include_in_prompt
                and self.persisted_slots == other.persisted_slots
                and self.run_pattern_completed == other.run_pattern_completed
            )
        return False

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
        from rasa.shared.core.flows.utils import extract_translations

        step_sequence = FlowStepSequence.from_json(flow_id, data.get("steps"))
        nlu_triggers = NLUTriggers.from_json(data.get("nlu_trigger"))

        if file_path and isinstance(file_path, Path):
            file_path = str(file_path)

        return Flow(
            id=flow_id,
            custom_name=data.get(KEY_NAME),
            description=data.get(KEY_DESCRIPTION),
            always_include_in_prompt=data.get(KEY_ALWAYS_INCLUDE_IN_PROMPT),
            # str or bool are permitted in the flow schema, but internally we want a str
            guard_condition=str(data[KEY_IF]) if KEY_IF in data else None,
            step_sequence=Flow.resolve_default_ids(step_sequence),
            nlu_triggers=nlu_triggers,
            # If we are reading the flows in after training the file_path is part of
            # data. When the model is trained, take the provided file_path.
            file_path=data.get(KEY_FILE_PATH) if KEY_FILE_PATH in data else file_path,
            persisted_slots=data.get(KEY_PERSISTED_SLOTS, []),
            run_pattern_completed=data.get(
                KEY_RUN_PATTERN_COMPLETED, DEFAULT_RUN_PATTERN_COMPLETED
            ),
            translation=extract_translations(
                translation_data=data.get(KEY_TRANSLATION, {})
            ),
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
            KEY_ID: self.id,
            KEY_STEPS: self.step_sequence.as_json(),
        }
        if self.custom_name is not None:
            data[KEY_NAME] = self.custom_name
        if self.description is not None:
            data[KEY_DESCRIPTION] = self.description
        if self.guard_condition is not None:
            data[KEY_IF] = self.guard_condition
        if self.always_include_in_prompt is not None:
            data[KEY_ALWAYS_INCLUDE_IN_PROMPT] = self.always_include_in_prompt
        if self.nlu_triggers:
            data[KEY_NLU_TRIGGER] = self.nlu_triggers.as_json()
        if self.file_path:
            data[KEY_FILE_PATH] = self.file_path
        if self.persisted_slots:
            data[KEY_PERSISTED_SLOTS] = self.persisted_slots
        if self.run_pattern_completed is not None:
            data[KEY_RUN_PATTERN_COMPLETED] = self.run_pattern_completed
        if self.translation:
            data[KEY_TRANSLATION] = {
                language_code: translation.dict()
                for language_code, translation in self.translation.items()
            }

        return data

    def localized_name(self, language: Optional[Language] = None) -> Optional[Text]:
        """Returns the language specific flow name or None.

        Args:
            language: Preferred language code.

        Returns:
            Flow name in the specified language or None.
        """
        language_code = language.code if language else None
        translation = self.translation.get(language_code)
        return translation.name if translation else None

    def readable_name(self, language: Optional[Language] = None) -> str:
        """Returns the flow's name in the specified language if available.

        Otherwise, falls back to the flow's name, and finally the flow's ID.

        Args:
            language: Preferred language code.

        Returns:
            string: the localized name, the default name, or the flow's ID.
        """
        return self.localized_name(language) or self.name or self.id

    def step_by_id(self, step_id: Optional[Text]) -> Optional[FlowStep]:
        """Returns the step with the given id."""
        if not step_id:
            return None

        if step_id == START_STEP:
            return StartFlowStep(self.id, self.first_step_in_flow().id)

        if step_id == END_STEP:
            return EndFlowStep(self.id)

        if step_id.startswith(CONTINUE_STEP_PREFIX):
            return ContinueFlowStep(self.id, step_id[len(CONTINUE_STEP_PREFIX) :])

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
        collect_steps: List[CollectInformationFlowStep] = []
        for step in self.steps_with_calls_resolved:
            # Only add collect steps that are not already in the list.
            # This is to avoid returning duplicate collect steps from called flows
            # in case the called flow is called multiple times.
            if (
                isinstance(step, CollectInformationFlowStep)
                and step not in collect_steps
            ):
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
                flow_id=self.id,
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
        from pypred.tiler import SimplePattern, tile

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
        all_paths = FlowPathsList(self.id, paths=[])
        start_step: FlowStep = self.first_step_in_flow()
        current_path: FlowPath = FlowPath(flow=self.id, nodes=[])
        visited_step_ids: Set[str] = set()

        self._go_over_steps(start_step, current_path, all_paths, visited_step_ids)

        structlogger.debug(
            "shared.core.flows.flow.extract_all_paths",
            comment="Extraction complete",
            number_of_paths=len(all_paths.paths),
            flow_name=self.name,
        )
        return all_paths

    def _go_over_steps(
        self,
        current_step: FlowStep,
        current_path: FlowPath,
        all_paths: FlowPathsList,
        visited_step_ids: Set[str],
        call_stack: Optional[
            List[Tuple[Optional[FlowStep], Optional[Flow], str]]
        ] = None,
    ) -> None:
        """Processes the flow steps recursively.

        Args:
            current_step: The current step being processed.
            current_path: The current path being constructed.
            all_paths: The list where completed paths are added.
            visited_step_ids: A set of steps that have been visited to avoid cycles.
            call_stack: Tuple list of (flow, path, flow_type) to track path when \
                calling flows through call and link steps.

        Returns:
            None: This function modifies all_paths in place by appending new paths
            as they are found.
        """
        if call_stack is None:
            call_stack = []

        # Check if the step is relevant for testable_paths extraction.
        # We only create new path nodes for CollectInformationFlowStep,
        # ActionFlowStep, CallFlowStep and LinkFlowStep,
        # because these are externally visible changes
        # in the assistant's behaviour (trackable in the e2e tests).
        # For other flow steps, we only follow their links.
        should_add_node = isinstance(
            current_step,
            (CollectInformationFlowStep, ActionFlowStep, CallFlowStep, LinkFlowStep),
        )
        if should_add_node:
            # Add current step to the current path that is being constructed.
            current_path.nodes.append(
                PathNode(
                    flow=current_path.flow,
                    step_id=current_step.id,
                    lines=current_step.metadata["line_numbers"],
                )
            )

        # Check if the current step has already been visited or
        # if the end of the path has been reached.
        # If so, and we’re not within a called flow, we terminate the current path.
        # This also applies for when we're inside a linked flow and reach its end.
        # If we're inside a called flow and reach its end,
        # continue with the next steps in its parent flow.
        if current_step.id in visited_step_ids or self.is_end_of_path(current_step):
            # Shallow copy is sufficient, since we only pop from the list and
            # don't mutate the objects inside the tuples.
            # The state of FlowStep and Flow does not change during the traversal.
            call_stack_copy = call_stack.copy()
            # parent_flow_type could be any of: None, i.e. main flow,
            # KEY_CALLED_FLOW(=called_flow) or KEY_LINKED_FLOW(=linked_flow)
            parent_step, parent_flow, parent_flow_type = (
                call_stack_copy.pop() if call_stack_copy else (None, None, None)
            )

            # Check if within a called flow.
            # If within linked flow, stop the traversal as this takes precedence.
            if parent_step and parent_flow_type == KEY_CALLED_FLOW:
                # As we have reached the END step of a called flow, we need to
                # continue with the next links of the parent step.
                if parent_flow is not None:
                    for link in parent_step.next.links:
                        parent_flow._handle_link(
                            current_path,
                            all_paths,
                            visited_step_ids,
                            link,
                            call_stack_copy,
                        )

            else:
                # Found a cycle, or reached an end step, do not proceed further.
                all_paths.paths.append(copy.deepcopy(current_path))

            # Backtrack: remove the last node after reaching a terminal step.
            # Ensures the path is correctly backtracked, after a path ends or
            # a cycle is detected.
            if should_add_node:
                current_path.nodes.pop()
            return

        # Mark current step as visited in this path.
        visited_step_ids.add(current_step.id)

        # If the current step is a call step, we need to resolve the call
        # and continue with the steps of the called flow.
        if isinstance(current_step, CallFlowStep):
            # Get the steps of the called flow and continue with them.
            called_flow = current_step.called_flow_reference
            if called_flow and (
                start_step_in_called_flow := called_flow.first_step_in_flow()
            ):
                call_stack.append((current_step, self, KEY_CALLED_FLOW))
                called_flow._go_over_steps(
                    start_step_in_called_flow,
                    current_path,
                    all_paths,
                    visited_step_ids,
                    call_stack,
                )

                # After processing the steps of the called (child) flow,
                # remove them from the visited steps
                # to allow the calling (parent) flow to revisit them later.
                visited_step_ids.remove(current_step.id)
                call_stack.pop()

                # Backtrack: remove the last node
                # after returning from a called (child) flow.
                # Ensures the parent flow can continue exploring other branches.
                if should_add_node:
                    current_path.nodes.pop()
            return

        # If the current step is a LinkFlowStep, step into the linked flow,
        # process its links, and do not return from that flow anymore.
        if isinstance(current_step, LinkFlowStep):
            # Get the steps of the linked flow and continue with them.
            linked_flow = current_step.linked_flow_reference
            if linked_flow and (
                start_step_in_linked_flow := linked_flow.first_step_in_flow()
            ):
                call_stack.append((current_step, self, KEY_LINKED_FLOW))
                linked_flow._go_over_steps(
                    start_step_in_linked_flow,
                    current_path,
                    all_paths,
                    visited_step_ids,
                    call_stack,
                )
                visited_step_ids.remove(current_step.id)
                call_stack.pop()

                # Backtrack: remove the last node
                # after returning from a linked (child) flow.
                # Ensures the parent can continue after the linked flow is processed.
                if should_add_node:
                    current_path.nodes.pop()
            return

        # Iterate over all links of the current step.
        for link in current_step.next.links:
            self._handle_link(
                current_path,
                all_paths,
                visited_step_ids,
                link,
                call_stack,
            )

        # Backtrack the current step and remove it from the path.
        visited_step_ids.remove(current_step.id)

        # Backtrack: remove the last node
        # after processing all links of the current step.
        # Ensures the next recursion can start once all links are explored.
        if should_add_node:
            current_path.nodes.pop()

    def _handle_link(
        self,
        current_path: FlowPath,
        all_paths: FlowPathsList,
        visited_step_ids: Set[str],
        link: FlowStepLink,
        call_stack: Optional[
            List[Tuple[Optional[FlowStep], Optional[Flow], str]]
        ] = None,
    ) -> None:
        """Handles the next step in a flow.

        Args:
            current_path: The current path being constructed.
            all_paths: The list where completed paths are added.
            visited_step_ids: A set of steps that have been visited to avoid cycles.
            link: The link to be followed.
            call_stack: Tuple list of (flow, path, flow_type) to track path when \
                calling flows through call and link steps..

        Returns:
            None: This function modifies all_paths in place by appending new paths
            as they are found.
        """
        # StaticFlowStepLink is a direct link to the next step.
        if isinstance(link, StaticFlowStepLink):
            # Find the step by its id and continue the path.
            if step := self._get_step_by_step_id(link.target_step_id):
                self._go_over_steps(
                    step,
                    current_path,
                    all_paths,
                    visited_step_ids,
                    call_stack,
                )
                return
        # IfFlowStepLink and ElseFlowStepLink are conditional links.
        elif isinstance(link, (IfFlowStepLink, ElseFlowStepLink)):
            if isinstance(link.target_reference, FlowStepSequence):
                # If the target is a FlowStepSequence, we need to go over all
                # child steps of the sequence.
                for child_step in link.target_reference.child_steps:
                    self._go_over_steps(
                        child_step,
                        current_path,
                        all_paths,
                        visited_step_ids,
                        call_stack,
                    )
                return
            else:
                # Find the step by its id and continue the path.
                if step := self._get_step_by_step_id(link.target_reference):
                    self._go_over_steps(
                        step,
                        current_path,
                        all_paths,
                        visited_step_ids,
                        call_stack,
                    )
                    return

    def is_end_of_path(self, step: FlowStep) -> bool:
        """Check if there is no path available from the current step."""
        if (
            len(step.next.links) == 1
            and isinstance(step.next.links[0], StaticFlowStepLink)
            and step.next.links[0].target == END_STEP
        ):
            return True
        return False

    def _get_step_by_step_id(
        self,
        step_id: Optional[str],
    ) -> Optional[FlowStep]:
        """Get a step by its id from a list of steps."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
