from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import errno
import inspect
import io
import json
import logging
import os
import requests

from requests.auth import HTTPBasicAuth

import rasa_core
import sys
from hashlib import sha1
from random import Random
from threading import Thread

import six
from builtins import input, range, str
from numpy import all, array
from typing import Text, Any, List, Optional, Tuple, Dict, Set

if six.PY2:
    from StringIO import StringIO
else:
    from io import StringIO

logger = logging.getLogger(__name__)


def configure_file_logging(loglevel, logfile):
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(loglevel)
        logging.getLogger('').addHandler(fh)
    logging.captureWarnings(True)


def add_logging_option_arguments(parser):
    """Add options to an argument parser to configure logging levels."""

    # arguments for logging configuration
    parser.add_argument(
            '--debug',
            help="Print lots of debugging statements. "
                 "Sets logging level to DEBUG",
            action="store_const",
            dest="loglevel",
            const=logging.DEBUG,
            default=logging.WARNING,
    )
    parser.add_argument(
            '-v', '--verbose',
            help="Be verbose. Sets logging level to INFO",
            action="store_const",
            dest="loglevel",
            const=logging.INFO,
    )


def class_from_module_path(module_path):
    # type: (Text) -> Any
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects. """
    import importlib

    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition('.')
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        return getattr(m, class_name)
    else:
        return globals()[module_path]


def module_path_from_instance(inst):
    # type: (Any) -> Text
    """Return the module path of an instances class."""
    return inst.__module__ + "." + inst.__class__.__name__


def dump_obj_as_json_to_file(filename, obj):
    # type: (Text, Any) -> None
    """Dump an object as a json string to a file."""

    dump_obj_as_str_to_file(filename, json.dumps(obj, indent=2))


def dump_obj_as_str_to_file(filename, text):
    # type: (Text, Text) -> None
    """Dump a text to a file."""

    with io.open(filename, 'w') as f:
        f.write(str(text))


def subsample_array(arr, max_values, can_modify_incoming_array=True, rand=None):
    # type: (List[Any], int, bool, Optional[Random]) -> List[Any]
    """Shuffles the array and returns `max_values` number of elements."""
    import random

    if not can_modify_incoming_array:
        arr = arr[:]
    if rand is not None:
        rand.shuffle(arr)
    else:
        random.shuffle(arr)
    return arr[:max_values]


def is_int(value):
    # type: (Any) -> bool
    """Checks if a value is an integer.

    The type of the value is not important, it might be an int or a float."""

    try:
        return value == int(value)
    except Exception:
        return False


def lazyproperty(fn):
    """Allows to avoid recomputing a property over and over.

    Instead the result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All succeeding calls
    will use the value stored in the private property."""

    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def create_dir_for_file(file_path):
    # type: (Text) -> None
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
        raise Exception("Can't create one hot. Index '{}' is out "
                        "of range (length '{}')".format(hot_idx, length))
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


def configure_colored_logging(loglevel):
    import coloredlogs
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles['asctime'] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles['debug'] = {}
    coloredlogs.install(
            level=loglevel,
            use_chroot=False,
            fmt='%(asctime)s %(levelname)-8s %(name)s  - %(message)s',
            level_styles=level_styles,
            field_styles=field_styles)


def request_input(valid_values=None, prompt=None, max_suggested=3):
    def wrong_input_message():
        print("Invalid answer, only {}{} allowed\n".format(
                ", ".join(valid_values[:max_suggested]),
                ",..." if len(valid_values) > max_suggested else ""))

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


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def wrap_with_color(text, color):
    return color + text + bcolors.ENDC


def print_color(text, color):
    print(wrap_with_color(text, color))


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
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view()).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        """Returns the encapsulated ndarray.

        If the wrapper is "tight", a copy of the encapsulated ndarray is
        returned. Otherwise, the encapsulated ndarray itself is returned."""

        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped


def fix_yaml_loader():
    """Ensure that any string read by yaml is represented as unicode."""
    import yaml
    import re

    def construct_yaml_str(self, node):
        # Override the default string handling function
        # to always return unicode objects
        return self.construct_scalar(node)

    # this will allow the reader to process emojis under py2
    # need to differentiate between narrow build (e.g. osx, windows) and
    # linux build. in the narrow build, emojis are 2 char strings using a
    # surrogate
    if sys.maxunicode == 0xffff:
        yaml.reader.Reader.NON_PRINTABLE = re.compile(
                '[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\ud83d\uE000-\uFFFD'
                '\ude00-\ude50]')
    else:
        yaml.reader.Reader.NON_PRINTABLE = re.compile(
                '[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\ud83d\uE000-\uFFFD'
                '\U00010000-\U0010FFFF]')

    yaml.Loader.add_constructor(u'tag:yaml.org,2002:str',
                                construct_yaml_str)
    yaml.SafeLoader.add_constructor(u'tag:yaml.org,2002:str',
                                    construct_yaml_str)


def read_yaml_file(filename):
    """Read contents of `filename` interpreting them as yaml."""
    return read_yaml_string(read_file(filename))


def read_yaml_string(string):
    if six.PY2:
        import yaml

        fix_yaml_loader()
        return yaml.load(string)
    else:
        import ruamel.yaml

        yaml_parser = ruamel.yaml.YAML(typ="safe")
        yaml_parser.version = "1.1"
        yaml_parser.unicode_supplementary = True

        return yaml_parser.load(string)


def _dump_yaml(obj, output):
    if six.PY2:
        import yaml

        yaml.safe_dump(obj, output,
                       default_flow_style=False,
                       allow_unicode=True)
    else:
        import ruamel.yaml

        yaml_writer = ruamel.yaml.YAML(pure=True, typ="safe")
        yaml_writer.unicode_supplementary = True
        yaml_writer.default_flow_style = False
        yaml_writer.version = "1.1"

        yaml_writer.dump(obj, output)


def dump_obj_as_yaml_to_file(filename, obj):
    """Writes data (python dict) to the filename in yaml repr."""
    with io.open(filename, 'w', encoding="utf-8") as output:
        _dump_yaml(obj, output)


def dump_obj_as_yaml_to_string(obj):
    """Writes data (python dict) to a yaml string."""
    str_io = StringIO()
    _dump_yaml(obj, str_io)
    return str_io.getvalue()


def read_file(filename, encoding="utf-8"):
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()


def is_training_data_empty(X):
    """Check if the training matrix does contain training samples."""
    return X.shape[0] == 0


def zip_folder(folder):
    """Create an archive from a folder."""
    import tempfile
    import shutil

    zipped_path = tempfile.NamedTemporaryFile(delete=False)
    zipped_path.close()

    # WARN: not thread save!
    return shutil.make_archive(zipped_path.name, str("zip"), folder)


def cap_length(s, char_limit=20, append_ellipsis=True):
    """Makes sure the string doesn't exceed the passed char limit.

    Appends an ellipsis if the string is to long."""

    if len(s) > char_limit:
        if append_ellipsis:
            return s[:char_limit - 3] + "..."
        else:
            return s[:char_limit]
    else:
        return s


def wait_for_threads(threads):
    # type: (List[Thread]) -> None
    """Block until all child threads have been terminated."""

    while len(threads) > 0:
        try:
            # Join all threads using a timeout so it doesn't block
            # Filter out threads which have been joined or are None
            [t.join(1000) for t in threads]
            threads = [t for t in threads if t.isAlive()]
        except KeyboardInterrupt:
            logger.info("Ctrl-c received! Sending kill to threads...")
            # It would be better at this point to properly shutdown every
            # thread (e.g. by setting a flag on it) Unfortunately, there
            # are IO operations that are blocking without a timeout
            # (e.g. sys.read) so threads that are waiting for one of
            # these calls can't check the set flag. Hence, we go the easy
            # route for now
            sys.exit(0)
    logger.info("Finished waiting for input threads to terminate. "
                "Stopping to serve forever.")


def extract_args(kwargs,  # type: Dict[Text, Any]
                 keys_to_extract  # type: Set[Text]
                 ):
    # type: (...) -> Tuple[Dict[Text, Any], Dict[Text, Any]]
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


def arguments_of(func):
    """Return the parameters of the function `func` as a list of their names."""

    try:
        # python 3.x is used
        return inspect.signature(func).parameters.keys()
    except AttributeError:
        # python 2.x is used
        return inspect.getargspec(func).args


def concat_url(base, subpath):
    # type: (Text, Optional[Text]) -> Text
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


def all_subclasses(cls):
    # type: (Any) -> List[Any]
    """Returns all known (imported) subclasses of a class."""

    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


def read_endpoint_config(filename, endpoint_type):
    # type: (Text, Text) -> Optional[rasa_core.utils.EndpointConfig]
    """Read an endpoint configuration file from disk and extract one config. """

    if not filename:
        return None

    content = read_yaml_file(filename)
    if endpoint_type in content:
        return EndpointConfig.from_dict(content[endpoint_type])
    else:
        return None


class EndpointConfig(object):
    """Configuration for an external HTTP endpoint."""

    def __init__(self, url, params=None, headers=None, basic_auth=None,
                 token=None, token_name="token"):
        self.url = url
        self.params = params if params else {}
        self.headers = headers if headers else {}
        self.basic_auth = basic_auth
        self.token = token
        self.token_name = token_name

    def request(self,
                method="post",  # type: Text
                subpath=None,  # type: Optional[Text]
                content_type="application/json",  # type: Text
                **kwargs  # type: Dict[Text, Any]
                ):
        """Send a HTTP request to the endpoint.

        All additional arguments will get passed through
        to `requests.request`."""

        # create the appropriate headers
        headers = self.headers.copy()
        if content_type:
            headers["Content-Type"] = content_type
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
            del kwargs["headers"]

        # create authentication parameters
        if self.basic_auth:
            auth = HTTPBasicAuth(self.basic_auth["username"],
                                 self.basic_auth["password"])
        else:
            auth = None

        url = concat_url(self.url, subpath)

        # construct GET parameters
        params = self.params.copy()

        # set the authentication token if present
        if self.token:
            params[self.token_name] = self.token

        if "params" in kwargs:
            params.update(kwargs["params"])
            del kwargs["params"]

        return requests.request(method,
                                url,
                                headers=headers,
                                params=params,
                                auth=auth,
                                **kwargs)

    @classmethod
    def from_dict(cls, data):
        return EndpointConfig(
                data.get("url"),
                data.get("params"),
                data.get("headers"),
                data.get("basic_auth"),
                data.get("token"),
                data.get("token_name"))

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return (other.url == self.url and
                    other.params == self.params and
                    other.headers == self.headers and
                    other.basic_auth == self.basic_auth and
                    other.token == self.token and
                    other.token_name == self.token_name)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
