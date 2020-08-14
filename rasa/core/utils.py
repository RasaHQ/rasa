import argparse
import json
import logging
import os
import re
import sys
from asyncio import Future
from decimal import Decimal
from hashlib import md5, sha1
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Text,
    Tuple,
    Union,
)

import aiohttp
import numpy as np
import rasa.utils.io as io_utils
from aiohttp import InvalidURL
from rasa.constants import (
    DEFAULT_SANIC_WORKERS,
    ENV_SANIC_WORKERS,
    DEFAULT_ENDPOINTS_PATH,
    YAML_VERSION,
)

# backwards compatibility 1.0.x
# noinspection PyUnresolvedReferences
from rasa.core.lock_store import LockStore, RedisLockStore
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from sanic import Sanic
from sanic.views import CompositionView
import rasa.cli.utils as cli_utils

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from random import Random


def configure_file_logging(
    logger_obj: logging.Logger, log_file: Optional[Text]
) -> None:
    """Configure logging to a file.

    Args:
        logger_obj: Logger object to configure.
        log_file: Path of log file to write to.
    """
    if not log_file:
        return

    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    file_handler = logging.FileHandler(log_file, encoding=io_utils.DEFAULT_ENCODING)
    file_handler.setLevel(logger_obj.level)
    file_handler.setFormatter(formatter)
    logger_obj.addHandler(file_handler)


def module_path_from_instance(inst: Any) -> Text:
    """Return the module path of an instance's class."""
    return inst.__module__ + "." + inst.__class__.__name__


def subsample_array(
    arr: List[Any],
    max_values: int,
    can_modify_incoming_array: bool = True,
    rand: Optional["Random"] = None,
) -> List[Any]:
    """Shuffles the array and returns `max_values` number of elements."""
    import random

    if not can_modify_incoming_array:
        arr = arr[:]
    if rand is not None:
        rand.shuffle(arr)
    else:
        random.shuffle(arr)
    return arr[:max_values]


def is_int(value: Any) -> bool:
    """Checks if a value is an integer.

    The type of the value is not important, it might be an int or a float."""

    # noinspection PyBroadException
    try:
        return value == int(value)
    except Exception:
        return False


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


def generate_id(prefix: Text = "", max_chars: Optional[int] = None) -> Text:
    """Generate a random UUID.

    Args:
        prefix: String to prefix the ID with.
        max_chars: Maximum number of characters.

    Returns:
        Generated random UUID.
    """
    import uuid

    gid = uuid.uuid4().hex
    if max_chars:
        gid = gid[:max_chars]

    return f"{prefix}{gid}"


# noinspection PyPep8Naming
class HashableNDArray:
    """Hashable wrapper for ndarray objects.

    Instances of ndarray are not hashable, meaning they cannot be added to
    sets, nor used as keys in dictionaries. This is by design - ndarray
    objects are mutable, and therefore cannot reliably implement the
    __hash__() method.

    The hashable class allows a way around this limitation. It implements
    the required methods for hashable objects in terms of an encapsulated
    ndarray object. This can be either a copied instance (which is safer)
    or the original object (which requires the user to be careful enough
    not to modify it)."""

    def __init__(self, wrapped, tight=False) -> None:
        """Creates a new hashable object encapsulating an ndarray.

        wrapped
            The wrapped ndarray.

        tight
            Optional. If True, a copy of the input ndaray is created.
            Defaults to False.
        """

        self.__tight = tight
        self.__wrapped = np.array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view()).hexdigest(), 16)

    def __eq__(self, other) -> bool:
        return np.all(self.__wrapped == other.__wrapped)

    def __hash__(self) -> int:
        return self.__hash

    def unwrap(self) -> np.ndarray:
        """Returns the encapsulated ndarray.

        If the wrapper is "tight", a copy of the encapsulated ndarray is
        returned. Otherwise, the encapsulated ndarray itself is returned."""

        if self.__tight:
            return np.array(self.__wrapped)

        return self.__wrapped


def _dump_yaml(obj: Dict, output: Union[Text, Path, StringIO]) -> None:
    import ruamel.yaml

    yaml_writer = ruamel.yaml.YAML(pure=True, typ="safe")
    yaml_writer.unicode_supplementary = True
    yaml_writer.default_flow_style = False
    yaml_writer.version = YAML_VERSION

    yaml_writer.dump(obj, output)


def dump_obj_as_yaml_to_file(
    filename: Union[Text, Path], obj: Any, should_preserve_key_order: bool = False
) -> None:
    """Writes `obj` to the filename in YAML repr.

    Args:
        filename: Target filename.
        obj: Object to dump.
        should_preserve_key_order: Whether to preserve key order in `obj`.
    """
    io_utils.write_yaml(
        obj, filename, should_preserve_key_order=should_preserve_key_order
    )


def dump_obj_as_yaml_to_string(obj: Dict) -> Text:
    """Writes data (python dict) to a yaml string."""
    str_io = StringIO()
    _dump_yaml(obj, str_io)
    return str_io.getvalue()


def list_routes(app: Sanic):
    """List all the routes of a sanic application.

    Mainly used for debugging."""
    from urllib.parse import unquote

    output = {}

    def find_route(suffix, path):
        for name, (uri, _) in app.router.routes_names.items():
            if name.split(".")[-1] == suffix and uri == path:
                return name
        return None

    for endpoint, route in app.router.routes_all.items():
        if endpoint[:-1] in app.router.routes_all and endpoint[-1] == "/":
            continue

        options = {}
        for arg in route.parameters:
            options[arg] = f"[{arg}]"

        if not isinstance(route.handler, CompositionView):
            handlers = [(list(route.methods)[0], route.name)]
        else:
            handlers = [
                (method, find_route(v.__name__, endpoint) or v.__name__)
                for method, v in route.handler.handlers.items()
            ]

        for method, name in handlers:
            line = unquote(f"{endpoint:50s} {method:30s} {name}")
            output[name] = line

    url_table = "\n".join(output[url] for url in sorted(output))
    logger.debug(f"Available web server routes: \n{url_table}")

    return output


def cap_length(s: Text, char_limit: int = 20, append_ellipsis: bool = True) -> Text:
    """Makes sure the string doesn't exceed the passed char limit.

    Appends an ellipsis if the string is too long."""

    if len(s) > char_limit:
        if append_ellipsis:
            return s[: char_limit - 3] + "..."
        else:
            return s[:char_limit]
    else:
        return s


def extract_args(
    kwargs: Dict[Text, Any], keys_to_extract: Set[Text]
) -> Tuple[Dict[Text, Any], Dict[Text, Any]]:
    """Go through the kwargs and filter out the specified keys.

    Return both, the filtered kwargs as well as the remaining kwargs."""

    remaining = {}
    extracted = {}
    for k, v in kwargs.items():
        if k in keys_to_extract:
            extracted[k] = v
        else:
            remaining[k] = v

    return extracted, remaining


def all_subclasses(cls: Any) -> List[Any]:
    """Returns all known (imported) subclasses of a class."""

    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in all_subclasses(s)
    ]


def is_limit_reached(num_messages: int, limit: int) -> bool:
    """Determine whether the number of messages has reached a limit.

    Args:
        num_messages: The number of messages to check.
        limit: Limit on the number of messages.

    Returns:
        `True` if the limit has been reached, otherwise `False`.
    """
    return limit is not None and num_messages >= limit


def read_lines(
    filename, max_line_limit=None, line_pattern=".*"
) -> Generator[Text, Any, None]:
    """Read messages from the command line and print bot responses."""

    line_filter = re.compile(line_pattern)

    with open(filename, "r", encoding=io_utils.DEFAULT_ENCODING) as f:
        num_messages = 0
        for line in f:
            m = line_filter.match(line)
            if m is not None:
                yield m.group(1 if m.lastindex else 0)
                num_messages += 1

            if is_limit_reached(num_messages, max_line_limit):
                break


def file_as_bytes(path: Text) -> bytes:
    """Read in a file as a byte array."""
    with open(path, "rb") as f:
        return f.read()


def convert_bytes_to_string(data: Union[bytes, bytearray, Text]) -> Text:
    """Convert `data` to string if it is a bytes-like object."""

    if isinstance(data, (bytes, bytearray)):
        return data.decode(io_utils.DEFAULT_ENCODING)

    return data


def get_file_hash(path: Text) -> Text:
    """Calculate the md5 hash of a file."""
    return md5(file_as_bytes(path)).hexdigest()


def get_text_hash(text: Text, encoding: Text = io_utils.DEFAULT_ENCODING) -> Text:
    """Calculate the md5 hash for a text."""
    return md5(text.encode(encoding)).hexdigest()


def get_dict_hash(data: Dict, encoding: Text = io_utils.DEFAULT_ENCODING) -> Text:
    """Calculate the md5 hash of a dictionary."""
    return md5(json.dumps(data, sort_keys=True).encode(encoding)).hexdigest()


async def download_file_from_url(url: Text) -> Text:
    """Download a story file from a url and persists it into a temp file.

    Returns the file path of the temp file that contains the
    downloaded content."""
    from rasa.nlu import utils as nlu_utils

    if not nlu_utils.is_url(url):
        raise InvalidURL(url)

    async with aiohttp.ClientSession() as session:
        async with session.get(url, raise_for_status=True) as resp:
            filename = io_utils.create_temporary_file(await resp.read(), mode="w+b")

    return filename


def remove_none_values(obj: Dict[Text, Any]) -> Dict[Text, Any]:
    """Remove all keys that store a `None` value."""
    return {k: v for k, v in obj.items() if v is not None}


def pad_lists_to_size(
    list_x: List, list_y: List, padding_value: Optional[Any] = None
) -> Tuple[List, List]:
    """Compares list sizes and pads them to equal length."""

    difference = len(list_x) - len(list_y)

    if difference > 0:
        return list_x, list_y + [padding_value] * difference
    elif difference < 0:
        return list_x + [padding_value] * (-difference), list_y
    else:
        return list_x, list_y


class AvailableEndpoints:
    """Collection of configured endpoints."""

    @classmethod
    def read_endpoints(cls, endpoint_file: Text) -> "AvailableEndpoints":
        nlg = read_endpoint_config(endpoint_file, endpoint_type="nlg")
        nlu = read_endpoint_config(endpoint_file, endpoint_type="nlu")
        action = read_endpoint_config(endpoint_file, endpoint_type="action_endpoint")
        model = read_endpoint_config(endpoint_file, endpoint_type="models")
        tracker_store = read_endpoint_config(
            endpoint_file, endpoint_type="tracker_store"
        )
        lock_store = read_endpoint_config(endpoint_file, endpoint_type="lock_store")
        event_broker = read_endpoint_config(endpoint_file, endpoint_type="event_broker")

        return cls(nlg, nlu, action, model, tracker_store, lock_store, event_broker)

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
        self.model = model
        self.action = action
        self.nlu = nlu
        self.nlg = nlg
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.event_broker = event_broker


def read_endpoints_from_path(
    endpoints_path: Union[Path, Text, None] = None
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


# noinspection PyProtectedMember
def set_default_subparser(parser, default_subparser) -> None:
    """default subparser selection. Call after setup, just before parse_args()

    parser: the name of the parser you're making changes to
    default_subparser: the name of the subparser to call by default"""
    subparser_found = False
    for arg in sys.argv[1:]:
        if arg in ["-h", "--help"]:  # global help if no subparser
            break
    else:
        for x in parser._subparsers._actions:
            if not isinstance(x, argparse._SubParsersAction):
                continue
            for sp_name in x._name_parser_map.keys():
                if sp_name in sys.argv[1:]:
                    subparser_found = True
        if not subparser_found:
            # insert default in first position before all other arguments
            sys.argv.insert(1, default_subparser)


def create_task_error_logger(error_message: Text = "") -> Callable[[Future], None]:
    """Error logger to be attached to a task.

    This will ensure exceptions are properly logged and won't get lost."""

    def handler(fut: Future) -> None:
        # noinspection PyBroadException
        try:
            fut.result()
        except Exception:
            logger.exception(
                "An exception was raised while running task. "
                "{}".format(error_message)
            )

    return handler


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


def _lock_store_is_redis_lock_store(
    lock_store: Union[EndpointConfig, LockStore, None]
) -> bool:
    if isinstance(lock_store, RedisLockStore):
        return True

    if isinstance(lock_store, LockStore):
        return False

    # `lock_store` is `None` or `EndpointConfig`
    return lock_store is not None and lock_store.type == "redis"


def number_of_sanic_workers(lock_store: Union[EndpointConfig, LockStore, None]) -> int:
    """Get the number of Sanic workers to use in `app.run()`.

    If the environment variable constants.ENV_SANIC_WORKERS is set and is not equal to
    1, that value will only be permitted if the used lock store supports shared
    resources across multiple workers (e.g. ``RedisLockStore``).
    """

    def _log_and_get_default_number_of_workers():
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

    if _lock_store_is_redis_lock_store(lock_store):
        logger.debug(f"Using {env_value} Sanic workers.")
        return env_value

    logger.debug(
        f"Unable to assign desired number of Sanic workers ({env_value}) as "
        f"no `RedisLockStore` endpoint configuration has been found."
    )
    return _log_and_get_default_number_of_workers()
