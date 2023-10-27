from __future__ import annotations
from dataclasses import dataclass
from typing import List, Generator, Any, Optional, Dict, Text, Set

import rasa.shared.utils.io
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.validation import validate_flow


@dataclass
class FlowsList:
    """A collection of flows.

    This class defines a number of methods that are executed across the available flows,
    such as fingerprinting (for retraining caching), collecting flows with
    specific attributes or collecting all utterances across all flows.
    """

    underlying_flows: List[Flow]
    """The flows contained in this FlowsList."""

    def __iter__(self) -> Generator[Flow, None, None]:
        """Iterates over the flows."""
        yield from self.underlying_flows

    def __len__(self):
        """Return the length of this FlowsList."""
        return len(self.underlying_flows)

    def is_empty(self) -> bool:
        """Returns whether the flows list is empty."""
        return len(self.underlying_flows) == 0

    @classmethod
    def from_json(cls, data: Optional[Dict[Text, Dict[Text, Any]]]) -> FlowsList:
        """Create a FlowsList object from serialized data

        Args:
            data: data for a FlowsList in a serialized format

        Returns:
            A FlowsList object.
        """
        if not data:
            return cls([])

        return cls(
            [
                Flow.from_json(flow_id, flow_config)
                for flow_id, flow_config in data.items()
            ]
        )

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
        return FlowsList(self.underlying_flows + other.underlying_flows)

    def flow_by_id(self, flow_id: Optional[Text]) -> Optional[Flow]:
        """Return the flow with the given id."""
        if not flow_id:
            return None

        for flow in self.underlying_flows:
            if flow.id == flow_id:
                return flow
        else:
            return None

    def validate(self) -> None:
        """Validate the flows."""
        for flow in self.underlying_flows:
            validate_flow(flow)

    @property
    def user_flow_ids(self) -> List[str]:
        """Get all ids of flows that can be started by a user.

        Returns:
            The ids of all flows that can be started by a user."""
        return [f.id for f in self.user_flows]

    @property
    def user_flows(self) -> FlowsList:
        """Get all flows that can be started by a user.

        Returns:
            All flows that can be started by a user."""
        return FlowsList(
            [f for f in self.underlying_flows if not f.is_rasa_default_flow]
        )

    @property
    def utterances(self) -> Set[str]:
        """Retrieve all utterances of all flows"""
        return set().union(*[flow.utterances for flow in self.underlying_flows])
