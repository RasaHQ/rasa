import logging
import os
from pathlib import Path
from socket import SOCK_DGRAM, SOCK_STREAM
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Text, Tuple, Union

import numpy as np
import structlog
from sanic import Sanic

import rasa.cli.utils as cli_utils
import rasa.shared.utils.io
from rasa.constants import DEFAULT_SANIC_WORKERS, ENV_SANIC_WORKERS
from rasa.core.constants import (
    ACTIVE_FLOW_METADATA_KEY,
    DOMAIN_GROUND_TRUTH_METADATA_KEY,
    STEP_ID_METADATA_KEY,
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.core.lock_store import InMemoryLockStore, LockStore, RedisLockStore
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH, TCP_PROTOCOL
from rasa.shared.core.constants import SlotMappingType
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import (
    EndpointConfig,
    read_endpoint_config,
    read_property_config_from_endpoints_file,
)
from rasa.utils.io import write_yaml

if TYPE_CHECKING:
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.flows.flows_list import FlowsList

structlogger = structlog.get_logger()


def configure_file_logging(
    logger_obj: logging.Logger,
    log_file: Optional[Text],
    use_syslog: Optional[bool],
    syslog_address: Optional[Text] = None,
    syslog_port: Optional[int] = None,
    syslog_protocol: Optional[Text] = None,
) -> None:
    """Configure logging to a file.

    Args:
        logger_obj: Logger object to configure.
        log_file: Path of log file to write to.
        use_syslog: Add syslog as a logger.
        syslog_address: Adress of the syslog server.
        syslog_port: Port of the syslog server.
        syslog_protocol: Protocol with the syslog server
    """
    if use_syslog:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] [%(process)d]" " %(message)s"
        )
        socktype = SOCK_STREAM if syslog_protocol == TCP_PROTOCOL else SOCK_DGRAM
        syslog_handler = logging.handlers.SysLogHandler(
            address=(syslog_address, syslog_port), socktype=socktype
        )
        syslog_handler.setLevel(logger_obj.level)
        syslog_handler.setFormatter(formatter)
        logger_obj.addHandler(syslog_handler)
    if log_file:
        formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler(
            log_file, encoding=rasa.shared.utils.io.DEFAULT_ENCODING
        )
        file_handler.setLevel(logger_obj.level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)


def one_hot(hot_idx: int, length: int, dtype: Optional[Text] = None) -> np.ndarray:
    """Create a one-hot array.

    Args:
        hot_idx: Index of the hot element.
        length: Length of the array.
        dtype: ``numpy.dtype`` of the array.

    Returns:
        One-hot array.
    """
    if hot_idx >= length:
        raise ValueError(
            "Can't create one hot. Index '{}' is out of range (length '{}')".format(
                hot_idx, length
            )
        )
    r = np.zeros(length, dtype)
    r[hot_idx] = 1
    return r


def dump_obj_as_yaml_to_file(
    filename: Union[Text, Path], obj: Any, should_preserve_key_order: bool = False
) -> None:
    """Writes `obj` to the filename in YAML repr.

    Args:
        filename: Target filename.
        obj: Object to dump.
        should_preserve_key_order: Whether to preserve key order in `obj`.
    """
    write_yaml(obj, filename, should_preserve_key_order=should_preserve_key_order)


def list_routes(app: Sanic) -> Dict[Text, Text]:
    """List all the routes of a sanic application. Mainly used for debugging."""
    from urllib.parse import unquote

    output = {}

    def find_route(suffix: Text, path: Text) -> Optional[Text]:
        for name, (uri, _) in app.router.routes_names.items():
            if name.split(".")[-1] == suffix and uri == path:
                return name
        return None

    for route in app.router.routes:
        endpoint = route.parts
        if endpoint[:-1] in app.router.routes_all and endpoint[-1] == "/":
            continue

        options = {}
        for arg in route._params:
            options[arg] = f"[{arg}]"

        name = route.name.replace("rasa_server.", "")
        methods = ",".join(route.methods)

        full_endpoint = "/" + "/".join(endpoint)
        line = unquote(f"{full_endpoint:50s} {methods:30s} {name}")
        output[name] = line

    url_table = "\n".join(output[url] for url in sorted(output))
    structlogger.debug(
        "server.routes", event_info=f"Available web server routes: \n{url_table}"
    )

    return output


def extract_args(
    kwargs: Dict[Text, Any], keys_to_extract: Set[Text]
) -> Tuple[Dict[Text, Any], Dict[Text, Any]]:
    """Go through the kwargs and filter out the specified keys.

    Return both, the filtered kwargs as well as the remaining kwargs.
    """
    remaining = {}
    extracted = {}
    for k, v in kwargs.items():
        if k in keys_to_extract:
            extracted[k] = v
        else:
            remaining[k] = v

    return extracted, remaining


def is_limit_reached(num_messages: int, limit: Optional[int]) -> bool:
    """Determine whether the number of messages has reached a limit.

    Args:
        num_messages: The number of messages to check.
        limit: Limit on the number of messages.

    Returns:
        `True` if the limit has been reached, otherwise `False`.
    """
    return limit is not None and num_messages >= limit


class AvailableEndpoints:
    """Collection of configured endpoints."""

    _instance = None

    @classmethod
    def read_endpoints(cls, endpoint_file: Text) -> "AvailableEndpoints":
        """Read the different endpoints from a yaml file."""
        nlg = read_endpoint_config(endpoint_file, endpoint_type="nlg")
        nlu = read_endpoint_config(endpoint_file, endpoint_type="nlu")
        action = read_endpoint_config(endpoint_file, endpoint_type="action_endpoint")
        model = read_endpoint_config(endpoint_file, endpoint_type="models")
        tracker_store = read_endpoint_config(
            endpoint_file, endpoint_type="tracker_store"
        )
        lock_store = read_endpoint_config(endpoint_file, endpoint_type="lock_store")
        event_broker = read_endpoint_config(endpoint_file, endpoint_type="event_broker")
        vector_store = read_endpoint_config(endpoint_file, endpoint_type="vector_store")
        model_groups = read_property_config_from_endpoints_file(
            endpoint_file, property_name="model_groups"
        )

        return cls(
            nlg,
            nlu,
            action,
            model,
            tracker_store,
            lock_store,
            event_broker,
            vector_store,
            model_groups,
        )

    def __init__(
        self,
        nlg: Optional[EndpointConfig] = None,
        nlu: Optional[EndpointConfig] = None,
        action: Optional[EndpointConfig] = None,
        model: Optional[EndpointConfig] = None,
        tracker_store: Optional[EndpointConfig] = None,
        lock_store: Optional[EndpointConfig] = None,
        event_broker: Optional[EndpointConfig] = None,
        vector_store: Optional[EndpointConfig] = None,
        model_groups: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Create an `AvailableEndpoints` object."""
        self.model = model
        self.action = action
        self.nlu = nlu
        self.nlg = nlg
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.event_broker = event_broker
        self.vector_store = vector_store
        self.model_groups = model_groups

    @classmethod
    def get_instance(
        cls, endpoint_file: Optional[Text] = DEFAULT_ENDPOINTS_PATH
    ) -> "AvailableEndpoints":
        """Get the singleton instance of AvailableEndpoints."""
        # Ensure that the instance is initialized only once.
        if cls._instance is None:
            cls._instance = cls.read_endpoints(endpoint_file)
        return cls._instance


def read_endpoints_from_path(
    endpoints_path: Optional[Union[Path, Text]] = None,
) -> AvailableEndpoints:
    """Get `AvailableEndpoints` object from specified path.

    Args:
        endpoints_path: Path of the endpoints file to be read. If `None` the
            default path for that file is used (`endpoints.yml`).

    Returns:
        `AvailableEndpoints` object read from endpoints file.

    """
    endpoints_config_path = cli_utils.get_validated_path(
        endpoints_path, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    return AvailableEndpoints.get_instance(endpoints_config_path)


def _lock_store_is_multi_worker_compatible(
    lock_store: Union[EndpointConfig, LockStore, None],
) -> bool:
    if isinstance(lock_store, InMemoryLockStore):
        return False

    if isinstance(lock_store, RedisLockStore):
        return True

    # `lock_store` is `None` or `EndpointConfig`
    return (
        lock_store is not None
        and isinstance(lock_store, EndpointConfig)
        and lock_store.type != "in_memory"
    )


def number_of_sanic_workers(lock_store: Union[EndpointConfig, LockStore, None]) -> int:
    """Get the number of Sanic workers to use in `app.run()`.

    If the environment variable constants.ENV_SANIC_WORKERS is set and is not equal to
    1, that value will only be permitted if the used lock store is not the
    `InMemoryLockStore`.
    """

    def _log_and_get_default_number_of_workers() -> int:
        structlogger.debug(
            "server.worker.set_count",
            number_of_workers=DEFAULT_SANIC_WORKERS,
            event_info=f"Using the default number of Sanic workers "
            f"({DEFAULT_SANIC_WORKERS}).",
        )
        return DEFAULT_SANIC_WORKERS

    try:
        env_value = int(os.environ.get(ENV_SANIC_WORKERS, DEFAULT_SANIC_WORKERS))
    except ValueError:
        structlogger.error(
            "server.worker.set_count.error",
            number_of_workers=os.environ[ENV_SANIC_WORKERS],
            event_info=f"Cannot convert environment variable `{ENV_SANIC_WORKERS}` "
            f"to int ('{os.environ[ENV_SANIC_WORKERS]}').",
        )
        return _log_and_get_default_number_of_workers()

    if env_value == DEFAULT_SANIC_WORKERS:
        return _log_and_get_default_number_of_workers()

    if env_value < 1:
        structlogger.warning(
            "server.worker.set_count.error_less_than_one",
            number_of_workers=env_value,
            event_info=f"Cannot set number of Sanic workers to the desired value "
            f"({env_value}). The number of workers must be at least 1.",
        )
        return _log_and_get_default_number_of_workers()

    if _lock_store_is_multi_worker_compatible(lock_store):
        structlogger.debug(
            "server.worker.set_count.success",
            event_info=f"Using {env_value} Sanic workers.",
            num_workers=env_value,
        )
        return env_value

    structlogger.warning(
        "server.worker.set_count.error_no_lock_store",
        event_info=f"Unable to assign desired number of Sanic workers ({env_value}) as "
        f"no `RedisLockStore` or custom `LockStore` endpoint "
        f"configuration has been found.",
        num_workers=env_value,
    )
    return _log_and_get_default_number_of_workers()


def add_bot_utterance_metadata(
    message: Dict[str, Any],
    domain_response_name: str,
    nlg: "NaturalLanguageGenerator",
    domain: "Domain",
    tracker: Optional[DialogueStateTracker],
) -> Dict[str, Any]:
    """Add metadata to the bot message."""
    message["utter_action"] = domain_response_name

    utter_source = message.get(UTTER_SOURCE_METADATA_KEY)
    if utter_source is None:
        utter_source = nlg.__class__.__name__
        message[UTTER_SOURCE_METADATA_KEY] = utter_source

    if tracker:
        message[ACTIVE_FLOW_METADATA_KEY] = tracker.active_flow
        message[STEP_ID_METADATA_KEY] = tracker.current_step_id

    if utter_source in ["IntentlessPolicy", "ContextualResponseRephraser"]:
        message[DOMAIN_GROUND_TRUTH_METADATA_KEY] = [
            response.get("text")
            for response in domain.responses.get(domain_response_name, [])
            if response.get("text") is not None
        ]

    return message


def should_force_slot_filling(
    tracker: Optional[DialogueStateTracker], flows: "FlowsList"
) -> Tuple[bool, Optional[str]]:
    """Check if the flow should force slot filling.

    This is only valid when the flow is at a collect information step which
    has set `force_slot_filling` to true and the slot has a valid `from_text` mapping.

    Args:
        tracker: The dialogue state tracker.
        flows: The list of flows.

    Returns:
        A tuple of a boolean indicating if the flow should force slot filling
        and the name of the slot if applicable.
    """
    from rasa.dialogue_understanding.processor.command_processor import (
        get_current_collect_step,
    )

    if tracker is None:
        structlogger.error(
            "slot.force_slot_filling.error",
            event_info="Tracker is None. Cannot force slot filling.",
        )
        return False, None

    stack = tracker.stack
    step = get_current_collect_step(stack, flows)
    if step is None or not step.force_slot_filling:
        return False, None

    slot_name = step.collect
    slot = tracker.slots.get(slot_name)

    if not slot:
        structlogger.debug(
            "slot.force_slot_filling.error",
            event_info=f"Slot '{slot_name}' not found in tracker. "
            f"Cannot force slot filling. "
            f"Please check if the slot is defined in the domain.",
        )
        return False, None

    for slot_mapping in slot.mappings:
        if slot_mapping.type == SlotMappingType.FROM_TEXT:
            return True, slot_name

    return False, None
