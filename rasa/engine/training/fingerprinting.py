import logging
from typing import Any, Dict, Text
from typing_extensions import Protocol, runtime_checkable

import rasa.shared.utils.io

logger = logging.getLogger(__name__)


@runtime_checkable
class Fingerprintable(Protocol):
    def fingerprint(self) -> Text:
        ...


def calculate_fingerprint_key(
    node_name: Text, config: Dict[Text, Any], inputs: Dict[Text, Fingerprintable]
):
    fingerprint_data = {
        "node_name": node_name,
        "config": config,
        "inputs": inputs,
    }

    fingerprint_key = rasa.shared.utils.io.deep_container_fingerprint(fingerprint_data)

    logger.debug(
        f"Calculated fingerprint_key: {fingerprint_key} for data {fingerprint_data}"
    )

    return fingerprint_key
