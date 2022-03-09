import inspect
import logging
from typing import Any, Dict, Text, Type
from typing_extensions import Protocol, runtime_checkable

import rasa.utils.common
import rasa.shared.utils.io

logger = logging.getLogger(__name__)


@runtime_checkable
class Fingerprintable(Protocol):
    """Interface that enforces training data can be fingerprinted."""

    def fingerprint(self) -> Text:
        """Returns a unique stable fingerprint of the data."""
        ...


def calculate_fingerprint_key(
    graph_component_class: Type,
    config: Dict[Text, Any],
    inputs: Dict[Text, Fingerprintable],
) -> Text:
    """Calculates a fingerprint key that uniquely represents a single node's execution.

    Args:
        graph_component_class: The graph component class.
        config: The component config.
        inputs: The inputs as a mapping of parent node name to input value.

    Returns:
        The fingerprint key.
    """
    fingerprint_data = {
        "node_name": rasa.utils.common.module_path_from_class(graph_component_class),
        "component_implementation": inspect.getsource(graph_component_class),
        "config": config,
        "inputs": inputs,
    }

    fingerprint_key = rasa.shared.utils.io.deep_container_fingerprint(fingerprint_data)

    logger.debug(
        f"Calculated fingerprint_key '{fingerprint_key}' for class "
        f"'{graph_component_class.__name__}'."
    )

    return fingerprint_key
