import logging
from typing import Any, Dict, Text
from typing_extensions import Protocol, runtime_checkable

import rasa.shared.utils.io

logger = logging.getLogger(__name__)


@runtime_checkable
class Fingerprintable(Protocol):
    """Interface that enforces training data can be fingerprinted."""

    def fingerprint(self) -> Text:
        """Returns a unique stable fingerprint of the data."""
        ...


def calculate_fingerprint_key(
    node_name: Text, config: Dict[Text, Any], inputs: Dict[Text, Fingerprintable]
) -> Text:
    """Calculates a fingerprint key that uniquely represents a single node's execution.

    Args:
        node_name: The name of the node.
        config: The component config.
        inputs: The inputs as a mapping of parent node name to input value.

    Returns:
        The fingerprint key.
    """
    fingerprint_data = {
        "node_name": node_name,
        "config": config,
        "inputs": inputs,
    }

    fingerprint_key = rasa.shared.utils.io.deep_container_fingerprint(fingerprint_data)

    logger.debug(
        f"Calculated fingerprint_key '{fingerprint_key}' for data '{fingerprint_data}'."
    )

    return fingerprint_key
