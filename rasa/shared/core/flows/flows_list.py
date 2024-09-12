from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Generator, Any, Optional, Dict, Text, Set, Union

import rasa.shared.utils.io
from rasa.shared.core.flows import Flow
from rasa.shared.core.flows.flow_path import FlowPathsList
from rasa.shared.core.flows.validation import (
    validate_flow,
    validate_link_in_call_restriction,
    validate_called_flows_exists,
    validate_linked_flows_exists,
    validate_nlu_trigger,
    validate_patterns_are_not_called_or_linked,
    validate_patterns_are_not_calling_or_linking_other_flows,
    validate_step_ids_are_unique,
)
from rasa.shared.core.slots import Slot


@dataclass
class FlowsList:
    """A collection of flows.

    This class defines a number of methods that are executed across the available flows,
    such as fingerprinting (for retraining caching), collecting flows with
    specific attributes or collecting all utterances across all flows.
    """

    underlying_flows: List[Flow]
    """The flows contained in this FlowsList."""

    def __post_init__(self) -> None:
        """Initializes the FlowsList object."""
        self._resolve_called_flows()

    def __iter__(self) -> Generator[Flow, None, None]:
        """Iterates over the flows."""
        yield from self.underlying_flows

    def __len__(self) -> int:
        """Return the length of this FlowsList."""
        return len(self.underlying_flows)

    def is_empty(self) -> bool:
        """Returns whether the flows list is empty."""
        return len(self.underlying_flows) == 0

    @classmethod
    def from_multiple_flows_lists(cls, *other: FlowsList) -> FlowsList:
        """Merges multiple lists of flows into a single flow ensuring each flow is
        unique, based on its ID.

        Args:
            other: Variable number of flow lists instances to be merged.

        Returns:
            Merged flow list.
        """
        merged_flows = dict()
        for flow_list in other:
            for flow in flow_list:
                if flow.id not in merged_flows:
                    merged_flows[flow.id] = flow
        return FlowsList(list(merged_flows.values()))

    @classmethod
    def from_json(
        cls,
        data: Optional[Dict[Text, Dict[Text, Any]]],
        file_path: Optional[Union[str, Path]] = None,
    ) -> FlowsList:
        """Create a FlowsList object from serialized data.

        Args:
            data: data for a FlowsList in a serialized format
            file_path: the file path of the flows

        Returns:
            A FlowsList object.
        """
        if not data:
            return cls(underlying_flows=[])

        return cls(
            underlying_flows=[
                Flow.from_json(flow_id, flow_config, file_path)
                for flow_id, flow_config in data.items()
            ]
        )

    def _resolve_called_flows(self) -> None:
        """Resolves the called flows."""
        from rasa.shared.core.flows.steps import CallFlowStep

        for flow in self.underlying_flows:
            for step in flow.steps:
                if isinstance(step, CallFlowStep) and not step.called_flow_reference:
                    # only resolve the reference, if it isn't already resolved
                    step.called_flow_reference = self.flow_by_id(step.call)

    def as_json_list(self) -> List[Dict[Text, Any]]:
        """Serialize the FlowsList object to list format and not to the original dict.

        Returns:
            The FlowsList object as serialized data in a list
        """
        return [flow.as_json() for flow in self.underlying_flows]

    def fingerprint(self) -> str:
        """Creates a fingerprint of the existing flows.

        Returns:
            The fingerprint of the flows.
        """
        flow_dicts = [flow.as_json() for flow in self.underlying_flows]
        return rasa.shared.utils.io.get_list_fingerprint(flow_dicts)

    def merge(self, other: FlowsList) -> FlowsList:
        """Merges two lists of flows together."""
        return FlowsList.from_multiple_flows_lists(self, other)

    def flow_by_id(self, flow_id: Text) -> Optional[Flow]:
        """Return the flow with the given id."""
        for flow in self.underlying_flows:
            if flow.id == flow_id:
                return flow
        else:
            return None

    def validate(self) -> None:
        """Validate the flows."""
        for flow in self.underlying_flows:
            validate_flow(flow)
        validate_nlu_trigger(self.underlying_flows)
        validate_link_in_call_restriction(self)
        validate_called_flows_exists(self)
        validate_linked_flows_exists(self)
        validate_patterns_are_not_called_or_linked(self)
        validate_patterns_are_not_calling_or_linking_other_flows(self)
        validate_step_ids_are_unique(self)

    @property
    def user_flow_ids(self) -> Set[str]:
        """Get all ids of flows that can be started by a user.

        Returns:
        The ids of all flows that can be started by a user.
        """
        return {f.id for f in self.user_flows}

    @property
    def flow_ids(self) -> Set[str]:
        """Get all ids of flows.

        Returns:
        The ids of all flows.
        """
        return {f.id for f in self.underlying_flows}

    @property
    def user_flows(self) -> FlowsList:
        """Get all flows that can be started by a user.

        Returns:
        All flows that can be started by a user.
        """
        return FlowsList(
            [f for f in self.underlying_flows if not f.is_rasa_default_flow]
        )

    @property
    def utterances(self) -> Set[str]:
        """Retrieve all utterances of all flows"""
        return {
            utterance for flow in self.underlying_flows for utterance in flow.utterances
        }

    def get_startable_flows(
        self,
        context: Optional[Dict[Text, Any]] = None,
        slots: Optional[Dict[Text, Slot]] = None,
    ) -> FlowsList:
        """Get all flows for which the starting conditions are met.

        Args:
            context: The context data to evaluate the starting conditions against.
            slots: The slots to evaluate the starting conditions against.

        Returns:
        All flows for which the starting conditions are met.
        """
        return FlowsList(
            [f for f in self.underlying_flows if f.is_startable(context, slots)]
        )

    def get_flows_always_included_in_prompt(self) -> FlowsList:
        """Gets all flows based on their inclusion status in prompts.

        Args:
            always_included: Inclusion status.

        Returns:
            All flows with the given inclusion status.
        """
        return FlowsList(
            [f for f in self.underlying_flows if f.always_include_in_prompt]
        )

    def exclude_link_only_flows(self) -> FlowsList:
        """Filter the given flows and exclude the flows that can
        be started only via link (flow guard evaluates to `False`).

        Returns:
            List of flows that doesn't contain flows that are
            only startable via link (another flow).
        """
        return FlowsList(
            [f for f in self.underlying_flows if not f.is_startable_only_via_link()]
        )

    def available_slot_names(self) -> Set[str]:
        """Get all slot names collected by flows."""
        return {
            step.collect
            for flow in self.underlying_flows
            for step in flow.get_collect_steps()
        }

    def available_custom_actions(self) -> Set[str]:
        """Get all custom actions collected by flows."""
        return set().union(*[flow.custom_actions for flow in self.underlying_flows])

    def extract_flow_paths(self) -> Dict[str, FlowPathsList]:
        paths = {}
        for flow in self.user_flows.underlying_flows:
            paths[flow.id] = flow.extract_all_paths()

        return paths
