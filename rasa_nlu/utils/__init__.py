from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import errno
import glob
import io
import json
import logging
import os
import re
import tempfile
from collections import namedtuple
from typing import List, Any
from typing import Optional
from typing import Text

import requests
import simplejson
import six
import yaml
from builtins import str
from future.utils import PY3
from requests.auth import HTTPBasicAuth


def add_logging_option_arguments(parser, default=logging.WARNING):
    """Add options to an argument parser to configure logging levels."""

    # arguments for logging configuration
    parser.add_argument(
            '--debug',
            help="Print lots of debugging statements. "
                 "Sets logging level to DEBUG",
            action="store_const",
            dest="loglevel",
            const=logging.DEBUG,
            default=default,
    )
    parser.add_argument(
            '-v', '--verbose',
            help="Be verbose. Sets logging level to INFO",
            action="store_const",
            dest="loglevel",
            const=logging.INFO,
    )


def relative_normpath(f, path):
    # type: (Optional[Text], Text) -> Optional[Text]
    """Return the path of file relative to `path`."""

    if f is not None:
        return os.path.normpath(os.path.relpath(f, path))
    else:
        return None


def create_dir(dir_path):
    # type: (Text) -> None
    """Creates a directory and its super paths.

    Succeeds even if the path already exists."""

    try:
        os.makedirs(dir_path)
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def create_dir_for_file(file_path):
    # type: (Text) -> None
    """Creates any missing parent directories of this files path."""

    try:
        os.makedirs(os.path.dirname(file_path))
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def list_directory(path):
    # type: (Text) -> List[Text]
    """Returns all files and folders excluding hidden files.

    If the path points to a file, returns the file. This is a recursive
    implementation returning files in any depth of the path."""

    if not isinstance(path, six.string_types):
        raise ValueError("Resourcename must be a string type")

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        results = []
        for base, dirs, files in os.walk(path):
            # remove hidden files
            goodfiles = filter(lambda x: not x.startswith('.'), files)
            results.extend(os.path.join(base, f) for f in goodfiles)
        return results
    else:
        raise ValueError("Could not locate the resource '{}'."
                         "".format(os.path.abspath(path)))


def list_files(path):
    # type: (Text) -> List[Text]
    """Returns all files excluding hidden files.

    If the path points to a file, returns the file."""

    return [fn for fn in list_directory(path) if os.path.isfile(fn)]


def list_subdirectories(path):
    # type: (Text) -> List[Text]
    """Returns all folders excluding hidden files.

    If the path points to a file, returns an empty list."""

    return [fn
            for fn in glob.glob(os.path.join(path, '*'))
            if os.path.isdir(fn)]


def lazyproperty(fn):
    """Allows to avoid recomputing a property over and over.

    The result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All
    succeeding calls will use the value stored in the private property."""

    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def list_to_str(l, delim=", ", quote="'"):
    return delim.join([quote + e + quote for e in l])


def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def module_path_from_object(o):
    """Returns the fully qualified class path of the instantiated object."""
    return o.__class__.__module__ + "." + o.__class__.__name__


def class_from_module_path(module_path):
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


def json_to_string(obj, **kwargs):
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def write_json_to_file(filename, obj, **kwargs):
    # type: (Text, Any) -> None
    """Write an object as a json string to a file."""

    write_to_file(filename, json_to_string(obj, **kwargs))


def write_to_file(filename, text):
    # type: (Text, Text) -> None
    """Write a text to a file."""

    with io.open(filename, 'w', encoding="utf-8") as f:
        f.write(str(text))


def read_file(filename, encoding="utf-8-sig"):
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()


def read_json_file(filename):
    """Read json from a file."""
    content = read_file(filename)
    try:
        return simplejson.loads(content)
    except ValueError as e:
        raise ValueError("Failed to read json from '{}'. Error: "
                         "{}".format(os.path.abspath(filename), e))


def fix_yaml_loader():
    """Ensure that any string read by yaml is represented as unicode."""
    from yaml import Loader, SafeLoader

    def construct_yaml_str(self, node):
        # Override the default string handling function
        # to always return unicode objects
        return self.construct_scalar(node)

    Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)
    SafeLoader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)


def replace_environment_variables():
    """Enable yaml loader to process the environment variables in the yaml."""
    import re
    import os

    env_var_pattern = re.compile(r"^(.*)\$\{(.*)\}(.*)$")
    yaml.add_implicit_resolver('!env_var', env_var_pattern)

    def env_var_constructor(loader, node):
        """Process environment variables found in the YAML."""
        value = loader.construct_scalar(node)
        prefix, env_var, postfix = env_var_pattern.match(value).groups()
        return prefix + os.environ[env_var] + postfix

    yaml.add_constructor(u'!env_var', env_var_constructor)


def read_yaml(content):
    fix_yaml_loader()
    replace_environment_variables()
    return yaml.load(content)


def read_yaml_file(filename):
    return read_yaml(read_file(filename, "utf-8"))


def build_entity(start, end, value, entity_type, **kwargs):
    """Builds a standard entity dictionary.

    Adds additional keyword parameters."""

    entity = {
        "start": start,
        "end": end,
        "value": value,
        "entity": entity_type
    }

    entity.update(kwargs)
    return entity


def is_model_dir(model_dir):
    """Checks if the given directory contains a model and can be safely removed.

    specifically checks if the directory has no subdirectories and
    if all files have an appropriate ending."""
    allowed_extensions = {".json", ".pkl", ".dat"}
    dir_tree = list(os.walk(model_dir))
    if len(dir_tree) != 1:
        return False
    model_dir, child_dirs, files = dir_tree[0]
    file_extenstions = [os.path.splitext(f)[1] for f in files]
    only_valid_files = all([ext in allowed_extensions
                            for ext in file_extenstions])
    return only_valid_files


def is_url(resource_name):
    """Return True if string is an http, ftp, or file URL path.

    This implementation is the same as the one used by matplotlib"""

    URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')
    return URL_REGEX.match(resource_name) is not None


def remove_model(model_dir):
    """Removes a model directory and all its content."""
    import shutil
    if is_model_dir(model_dir):
        shutil.rmtree(model_dir)
        return True
    else:
        raise ValueError("Cannot remove {}, it seems it is not a model "
                         "directory".format(model_dir))


def as_text_type(t):
    if isinstance(t, six.text_type):
        return t
    else:
        return six.text_type(t)


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


def pycloud_unpickle(file_name):
    # type: (Text) -> Any
    """Unpickle an object from file using cloudpickle."""
    from future.utils import PY2
    import cloudpickle

    with io.open(file_name, 'rb') as f:  # pragma: no test
        if PY2:
            return cloudpickle.load(f)
        else:
            return cloudpickle.load(f, encoding="latin-1")


def pycloud_pickle(file_name, obj):
    # type: (Text, Any) -> None
    """Pickle an object to a file using cloudpickle."""
    import cloudpickle

    with io.open(file_name, 'wb') as f:
        cloudpickle.dump(obj, f)


def create_temporary_file(data, suffix="", mode="w+"):
    """Creates a tempfile.NamedTemporaryFile object for data.

    mode defines NamedTemporaryFile's  mode parameter in py3."""

    if PY3:
        f = tempfile.NamedTemporaryFile(mode=mode, suffix=suffix,
                                        delete=False)
        f.write(data)
    else:
        f = tempfile.NamedTemporaryFile("w+", suffix=suffix,
                                        delete=False)
        f.write(data.encode("utf-8"))

    f.close()
    return f.name


def zip_folder(folder):
    """Create an archive from a folder."""
    import tempfile
    import shutil

    zipped_path = tempfile.NamedTemporaryFile(delete=False)
    zipped_path.close()

    # WARN: not thread save!
    return shutil.make_archive(zipped_path.name, str("zip"), folder)


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


def read_endpoint_config(filename, endpoint_type):
    # type: (Text, Text) -> Optional[EndpointConfig]
    """Read an endpoint configuration file from disk and extract one

    config. """
    if not filename:
        return None

    content = read_yaml_file(filename)
    if endpoint_type in content:
        return EndpointConfig.from_dict(content[endpoint_type])
    else:
        return None


def read_endpoints(endpoint_file):
    model = read_endpoint_config(endpoint_file,
                                 endpoint_type="model")
    data = read_endpoint_config(endpoint_file,
                                endpoint_type="data")

    return AvailableEndpoints(model, data)


# The EndpointConfig class is currently used to define external endpoints
# for pulling NLU models from a server and training data
AvailableEndpoints = namedtuple('AvailableEndpoints', 'model data')


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
                content_type="application/json",  # type: Optional[Text]
                **kwargs  # type: Any
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
