# -*- coding: utf-8 -*-
import argparse
import asyncio
import errno
import json
import logging
import os
import re
import sys
import tarfile
import tempfile
import warnings
import zipfile
from asyncio import AbstractEventLoop, Future
from hashlib import md5, sha1
from io import BytesIO as IOReader, StringIO
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Text,
    Tuple,
    Callable,
    Awaitable,
)

import aiohttp
from aiohttp import InvalidURL
from requests.exceptions import InvalidURL
from sanic import Sanic
from sanic.request import Request
from sanic.views import CompositionView

import rasa.utils.endpoints
from rasa.utils.endpoints import read_endpoint_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from random import Random


def configure_file_logging(loglevel, logfile):
    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(loglevel)
        logging.getLogger("").addHandler(fh)
    logging.captureWarnings(True)


# noinspection PyUnresolvedReferences
def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Any:
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects. """
    import importlib

    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        return getattr(m, class_name)
    else:
        module = globals().get(module_path, locals().get(module_path))
        if module is not None:
            return module

        if lookup_path:
            # last resort: try to import the class from the lookup path
            m = importlib.import_module(lookup_path)
            return getattr(m, module_path)
        else:
            raise ImportError("Cannot retrieve class from path {}.".format(module_path))


def module_path_from_instance(inst: Any) -> Text:
    """Return the module path of an instance's class."""
    return inst.__module__ + "." + inst.__class__.__name__


def dump_obj_as_json_to_file(filename: Text, obj: Any) -> None:
    """Dump an object as a json string to a file."""

    dump_obj_as_str_to_file(filename, json.dumps(obj, indent=2))


def dump_obj_as_str_to_file(filename: Text, text: Text) -> None:
    """Dump a text to a file."""

    with open(filename, "w", encoding="utf-8") as f:
        # noinspection PyTypeChecker
        f.write(str(text))


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


def lazyproperty(fn):
    """Allows to avoid recomputing a property over and over.

    Instead the result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All succeeding calls
    will use the value stored in the private property."""

    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def create_dir_for_file(file_path: Text) -> None:
    """Creates any missing parent directories of this files path."""

    try:
        os.makedirs(os.path.dirname(file_path))
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def one_hot(hot_idx, length, dtype=None):
    import numpy

    if hot_idx >= length:
        raise ValueError(
            "Can't create one hot. Index '{}' is out "
            "of range (length '{}')".format(hot_idx, length)
        )
    r = numpy.zeros(length, dtype)
    r[hot_idx] = 1
    return r


def str_range_list(start, end):
    return [str(e) for e in range(start, end)]


def generate_id(prefix="", max_chars=None):
    import uuid

    gid = uuid.uuid4().hex
    if max_chars:
        gid = gid[:max_chars]

    return "{}{}".format(prefix, gid)


def request_input(valid_values=None, prompt=None, max_suggested=3):
    def wrong_input_message():
        print (
            "Invalid answer, only {}{} allowed\n".format(
                ", ".join(valid_values[:max_suggested]),
                ",..." if len(valid_values) > max_suggested else "",
            )
        )

    while True:
        try:
            input_value = input(prompt) if prompt else input()
            if valid_values is not None and input_value not in valid_values:
                wrong_input_message()
                continue
        except ValueError:
            wrong_input_message()
            continue
        return input_value


# noinspection PyPep8Naming


class HashableNDArray(object):
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

    def __init__(self, wrapped, tight=False):
        """Creates a new hashable object encapsulating an ndarray.

        wrapped
            The wrapped ndarray.

        tight
            Optional. If True, a copy of the input ndaray is created.
            Defaults to False.
        """
        from numpy import array

        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view()).hexdigest(), 16)

    def __eq__(self, other):
        from numpy import all

        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        """Returns the encapsulated ndarray.

        If the wrapper is "tight", a copy of the encapsulated ndarray is
        returned. Otherwise, the encapsulated ndarray itself is returned."""
        from numpy import array

        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped


def _dump_yaml(obj, output):
    import ruamel.yaml

    yaml_writer = ruamel.yaml.YAML(pure=True, typ="safe")
    yaml_writer.unicode_supplementary = True
    yaml_writer.default_flow_style = False
    yaml_writer.version = "1.1"

    yaml_writer.dump(obj, output)


def dump_obj_as_yaml_to_file(filename, obj):
    """Writes data (python dict) to the filename in yaml repr."""
    with open(filename, "w", encoding="utf-8") as output:
        _dump_yaml(obj, output)


def dump_obj_as_yaml_to_string(obj):
    """Writes data (python dict) to a yaml string."""
    str_io = StringIO()
    _dump_yaml(obj, str_io)
    return str_io.getvalue()


def read_json_file(filename):
    """Read json from a file"""
    with open(filename) as f:
        return json.load(f)


def list_routes(app: Sanic):
    """List all the routes of a sanic application.

    Mainly used for debugging."""
    from urllib.parse import unquote

    output = {}

    def find_route(suffix, path):
        for name, (uri, _) in app.router.routes_names.items():
            if name.endswith(suffix) and uri == path:
                return name
        return None

    for endpoint, route in app.router.routes_all.items():
        if endpoint[:-1] in app.router.routes_all and endpoint[-1] == "/":
            continue

        options = {}
        for arg in route.parameters:
            options[arg] = "[{0}]".format(arg)

        if not isinstance(route.handler, CompositionView):
            handlers = [(list(route.methods)[0], route.name)]
        else:
            handlers = [
                (method, find_route(v.__name__, endpoint) or v.__name__)
                for method, v in route.handler.handlers.items()
            ]

        for method, name in handlers:
            line = unquote("{:50s} {:30s} {}".format(endpoint, method, name))
            output[name] = line

    url_table = "\n".join(output[url] for url in sorted(output))
    logger.debug("Available web server routes: \n{}".format(url_table))

    return output


def zip_folder(folder):
    """Create an archive from a folder."""
    import shutil

    zipped_path = tempfile.NamedTemporaryFile(delete=False)
    zipped_path.close()

    # WARN: not thread save!
    return shutil.make_archive(zipped_path.name, str("zip"), folder)


def unarchive(byte_array: bytes, directory: Text) -> Text:
    """Tries to unpack a byte array interpreting it as an archive.

    Tries to use tar first to unpack, if that fails, zip will be used."""

    try:
        tar = tarfile.open(fileobj=IOReader(byte_array))
        tar.extractall(directory)
        tar.close()
        return directory
    except tarfile.TarError:
        zip_ref = zipfile.ZipFile(IOReader(byte_array))
        zip_ref.extractall(directory)
        zip_ref.close()
        return directory


def cap_length(s, char_limit=20, append_ellipsis=True):
    """Makes sure the string doesn't exceed the passed char limit.

    Appends an ellipsis if the string is to long."""

    if len(s) > char_limit:
        if append_ellipsis:
            return s[: char_limit - 3] + "..."
        else:
            return s[:char_limit]
    else:
        return s


def write_request_body_to_file(request: Request, path: Text):
    """Writes the body of `request` to `path`."""

    with open(path, "w+b") as f:
        f.write(request.body)


def bool_arg(request: Request, name: Text, default: bool = True) -> bool:
    """Return a passed boolean argument of the request or a default.

    Checks the `name` parameter of the request if it contains a valid
    boolean value. If not, `default` is returned."""

    return request.args.get(name, str(default)).lower() == "true"


def float_arg(
    request: Request, key: Text, default: Optional[float] = None
) -> Optional[float]:
    """Return a passed argument cast as a float or None.

    Checks the `name` parameter of the request if it contains a valid
    float value. If not, `None` is returned."""

    arg = request.args.get(key, default)

    if arg is default:
        return arg

    try:
        return float(str(arg))
    except (ValueError, TypeError):
        logger.warning("Failed to convert '{}' to float.".format(arg))
        return default


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


def concat_url(base: Text, subpath: Optional[Text]) -> Text:
    """Append a subpath to a base url.

    Strips leading slashes from the subpath if necessary. This behaves
    differently than `urlparse.urljoin` and will not treat the subpath
    as a base url if it starts with `/` but will always append it to the
    `base`."""

    if subpath:
        url = base
        if not base.endswith("/"):
            url += "/"
        if subpath.startswith("/"):
            subpath = subpath[1:]
        return url + subpath
    else:
        return base


def all_subclasses(cls: Any) -> List[Any]:
    """Returns all known (imported) subclasses of a class."""

    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in all_subclasses(s)
    ]


def is_limit_reached(num_messages, limit):
    return limit is not None and num_messages >= limit


def read_lines(filename, max_line_limit=None, line_pattern=".*"):
    """Read messages from the command line and print bot responses."""

    line_filter = re.compile(line_pattern)

    with open(filename, "r", encoding="utf-8") as f:
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


def get_file_hash(path: Text) -> Text:
    """Calculate the md5 hash of a file."""
    return md5(file_as_bytes(path)).hexdigest()


def get_text_hash(text: Text, encoding: Text = "utf-8") -> Text:
    """Calculate the md5 hash of a file."""
    return md5(text.encode(encoding)).hexdigest()


async def download_file_from_url(url: Text) -> Text:
    """Download a story file from a url and persists it into a temp file.

    Returns the file path of the temp file that contains the
    downloaded content."""
    from rasa.nlu import utils as nlu_utils

    if not nlu_utils.is_url(url):
        raise InvalidURL(url)

    async with aiohttp.ClientSession() as session:
        async with session.get(url, raise_for_status=True) as resp:
            filename = nlu_utils.create_temporary_file(await resp.read(), mode="w+b")

    return filename


def remove_none_values(obj: Dict[Text, Any]) -> Dict[Text, Any]:
    """Remove all keys that store a `None` value."""
    return {k: v for k, v in obj.items() if v is not None}


def pad_list_to_size(_list, size, padding_value=None):
    """Pads _list with padding_value up to size"""
    return _list + [padding_value] * (size - len(_list))


class AvailableEndpoints(object):
    """Collection of configured endpoints."""

    @classmethod
    def read_endpoints(cls, endpoint_file):
        nlg = read_endpoint_config(endpoint_file, endpoint_type="nlg")
        nlu = read_endpoint_config(endpoint_file, endpoint_type="nlu")
        action = read_endpoint_config(endpoint_file, endpoint_type="action_endpoint")
        model = read_endpoint_config(endpoint_file, endpoint_type="models")
        tracker_store = read_endpoint_config(
            endpoint_file, endpoint_type="tracker_store"
        )
        event_broker = read_endpoint_config(endpoint_file, endpoint_type="event_broker")

        return cls(nlg, nlu, action, model, tracker_store, event_broker)

    def __init__(
        self,
        nlg=None,
        nlu=None,
        action=None,
        model=None,
        tracker_store=None,
        event_broker=None,
    ):
        self.model = model
        self.action = action
        self.nlu = nlu
        self.nlg = nlg
        self.tracker_store = tracker_store
        self.event_broker = event_broker


# noinspection PyProtectedMember
def set_default_subparser(parser, default_subparser):
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


class LockCounter(asyncio.Lock):
    """Decorated asyncio lock that counts how many coroutines are waiting.

    The counter can be used to discard the lock when there is no coroutine
    waiting for it. For this to work, there should not be any execution yield
    between retrieving the lock and acquiring it, otherwise there might be
    race conditions."""

    def __init__(self) -> None:
        super().__init__()
        self.wait_counter = 0

    async def acquire(self) -> bool:
        """Acquire the lock, makes sure only one coroutine can retrieve it."""

        self.wait_counter += 1
        try:
            return await super(LockCounter, self).acquire()  # type: ignore
        finally:
            self.wait_counter -= 1

    def is_someone_waiting(self) -> bool:
        """Check if a coroutine is waiting for this lock to be freed."""
        return self.wait_counter != 0
