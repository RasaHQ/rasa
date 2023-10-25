from __future__ import annotations

from typing import List, Generator, Any, Optional, Dict, Text, Set

import rasa.shared
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.validation import validate_flow


class FlowsList:
    """Represents the configuration of a list of flow.

    We need this class to be able to fingerprint the flows configuration.
    Fingerprinting is needed to make sure that the model is retrained if the
    flows configuration changes.
    """

    def __init__(self, flows: List[Flow]) -> None:
        """Initializes the configuration of flows.

        Args:
            flows: The flows to be configured.
        """
        self.underlying_flows = flows

    def __iter__(self) -> Generator[Flow, None, None]:
        """Iterates over the flows."""
        yield from self.underlying_flows

    def __eq__(self, other: Any) -> bool:
        """Compares the flows."""
        return (
            isinstance(other, FlowsList)
            and self.underlying_flows == other.underlying_flows
        )

    def is_empty(self) -> bool:
        """Returns whether the flows list is empty."""
        return len(self.underlying_flows) == 0

    @classmethod
    def from_json(
        cls, flows_configs: Optional[Dict[Text, Dict[Text, Any]]]
    ) -> FlowsList:
        """Used to read flows from parsed YAML.

        Args:
            flows_configs: The parsed YAML as a dictionary.

        Returns:
            The parsed flows.
        """
        if not flows_configs:
            return cls([])

        return cls(
            [
                Flow.from_json(flow_id, flow_config)
                for flow_id, flow_config in flows_configs.items()
            ]
        )

    def as_json(self) -> List[Dict[Text, Any]]:
        """Returns the flows as a dictionary.

        Returns:
            The flows as a dictionary.
        """
        return [flow.as_json() for flow in self.underlying_flows]

    def fingerprint(self) -> str:
        """Creates a fingerprint of the flows configuration.

        Returns:
            The fingerprint of the flows configuration.
        """
        flow_dicts = [flow.as_json() for flow in self.underlying_flows]
        return rasa.shared.utils.io.get_list_fingerprint(flow_dicts)

    def merge(self, other: FlowsList) -> FlowsList:
        """Merges two lists of flows together."""
        return FlowsList(self.underlying_flows + other.underlying_flows)

    def flow_by_id(self, id: Optional[Text]) -> Optional[Flow]:
        """Return the flow with the given id."""
        if not id:
            return None

        for flow in self.underlying_flows:
            if flow.id == id:
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
