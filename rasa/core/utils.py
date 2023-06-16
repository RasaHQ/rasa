import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Set, Text, Tuple, Union

import numpy as np

import rasa.shared.utils.io
from rasa.constants import DEFAULT_SANIC_WORKERS, ENV_SANIC_WORKERS
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH, TCP_PROTOCOL

from rasa.core.lock_store import LockStore, RedisLockStore, InMemoryLockStore
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from sanic import Sanic
from socket import SOCK_DGRAM, SOCK_STREAM
import rasa.cli.utils as cli_utils


logger = logging.getLogger(__name__)


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
            "Can't create one hot. Index '{}' is out "
            "of range (length '{}')".format(hot_idx, length)
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
    rasa.shared.utils.io.write_yaml(
        obj, filename, should_preserve_key_order=should_preserve_key_order
    )


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

        handlers = [(list(route.methods)[0], route.name.replace("rasa_server.", ""))]

        for method, name in handlers:
            full_endpoint = "/" + "/".join(endpoint)
            line = unquote(f"{full_endpoint:50s} {method:30s} {name}")
            output[name] = line

    url_table = "\n".join(output[url] for url in sorted(output))
    logger.debug(f"Available web server routes: \n{url_table}")

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


def file_as_bytes(path: Text) -> bytes:
    """Read in a file as a byte array."""
    with open(path, "rb") as f:
        return f.read()


class AvailableEndpoints:
    """Collection of configured endpoints."""

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

        return cls(
            nlg,
            nlu,
            action,
            model,
            tracker_store,
            lock_store,
            event_broker,
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
    ) -> None:
        """Create an `AvailableEndpoints` object."""
        self.model = model
        self.action = action
        self.nlu = nlu
        self.nlg = nlg
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.event_broker = event_broker


def read_endpoints_from_path(
    endpoints_path: Optional[Union[Path, Text]] = None
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
    return AvailableEndpoints.read_endpoints(endpoints_config_path)


def replace_floats_with_decimals(obj: Any, round_digits: int = 9) -> Any:
    """Convert all instances in `obj` of `float` to `Decimal`.

    Args:
        obj: Input object.
        round_digits: Rounding precision of `Decimal` values.

    Returns:
        Input `obj` with all `float` types replaced by `Decimal`s rounded to
        `round_digits` decimal places.
    """

    def _float_to_rounded_decimal(s: Text) -> Decimal:
        return Decimal(s).quantize(Decimal(10) ** -round_digits)

    return json.loads(json.dumps(obj), parse_float=_float_to_rounded_decimal)


class DecimalEncoder(json.JSONEncoder):
    """`json.JSONEncoder` that dumps `Decimal`s as `float`s."""

    def default(self, obj: Any) -> Any:
        """Get serializable object for `o`.

        Args:
            obj: Object to serialize.

        Returns:
            `obj` converted to `float` if `o` is a `Decimals`, else the base class
            `default()` method.
        """
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def replace_decimals_with_floats(obj: Any) -> Any:
    """Convert all instances in `obj` of `Decimal` to `float`.

    Args:
        obj: A `List` or `Dict` object.

    Returns:
        Input `obj` with all `Decimal` types replaced by `float`s.
    """
    return json.loads(json.dumps(obj, cls=DecimalEncoder))


def _lock_store_is_multi_worker_compatible(
    lock_store: Union[EndpointConfig, LockStore, None]
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
        logger.debug(
            f"Using the default number of Sanic workers ({DEFAULT_SANIC_WORKERS})."
        )
        return DEFAULT_SANIC_WORKERS

    try:
        env_value = int(os.environ.get(ENV_SANIC_WORKERS, DEFAULT_SANIC_WORKERS))
    except ValueError:
        logger.error(
            f"Cannot convert environment variable `{ENV_SANIC_WORKERS}` "
            f"to int ('{os.environ[ENV_SANIC_WORKERS]}')."
        )
        return _log_and_get_default_number_of_workers()

    if env_value == DEFAULT_SANIC_WORKERS:
        return _log_and_get_default_number_of_workers()

    if env_value < 1:
        logger.debug(
            f"Cannot set number of Sanic workers to the desired value "
            f"({env_value}). The number of workers must be at least 1."
        )
        return _log_and_get_default_number_of_workers()

    if _lock_store_is_multi_worker_compatible(lock_store):
        logger.debug(f"Using {env_value} Sanic workers.")
        return env_value

    logger.debug(
        f"Unable to assign desired number of Sanic workers ({env_value}) as "
        f"no `RedisLockStore` or custom `LockStore` endpoint "
        f"configuration has been found."
    )
    return _log_and_get_default_number_of_workers()
