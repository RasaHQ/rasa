import errno
import glob
import io
import json
import logging
import os
import re
import tempfile
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Text, Type

import simplejson

import rasa.utils.io

from rasa.utils.endpoints import read_endpoint_config


def add_logging_option_arguments(parser, default=logging.WARNING):
    """Add options to an argument parser to configure logging levels."""

    # arguments for logging configuration
    parser.add_argument(
        "--debug",
        help="Print lots of debugging statements. Sets logging level to DEBUG",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=default,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose. Sets logging level to INFO",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )


def relative_normpath(f: Optional[Text], path: Text) -> Optional[Text]:
    """Return the path of file relative to `path`."""

    if f is not None:
        return os.path.normpath(os.path.relpath(f, path))
    else:
        return None


def create_dir(dir_path: Text) -> None:
    """Creates a directory and its super paths.

    Succeeds even if the path already exists."""

    try:
        os.makedirs(dir_path)
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def create_dir_for_file(file_path: Text) -> None:
    """Creates any missing parent directories of this files path."""

    try:
        os.makedirs(os.path.dirname(file_path))
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def list_directory(path: Text) -> List[Text]:
    """Returns all files and folders excluding hidden files.

    If the path points to a file, returns the file. This is a recursive
    implementation returning files in any depth of the path."""

    if not isinstance(path, str):
        raise ValueError(
            "`resource_name` must be a string type. "
            "Got `{}` instead".format(type(path))
        )

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        results = []
        for base, dirs, files in os.walk(path):
            # remove hidden files
            goodfiles = filter(lambda x: not x.startswith("."), files)
            results.extend(os.path.join(base, f) for f in goodfiles)
        return results
    else:
        raise ValueError(
            "Could not locate the resource '{}'.".format(os.path.abspath(path))
        )


def list_files(path: Text) -> List[Text]:
    """Returns all files excluding hidden files.

    If the path points to a file, returns the file."""

    return [fn for fn in list_directory(path) if os.path.isfile(fn)]


def list_subdirectories(path: Text) -> List[Text]:
    """Returns all folders excluding hidden files.

    If the path points to a file, returns an empty list."""

    return [fn for fn in glob.glob(os.path.join(path, "*")) if os.path.isdir(fn)]


def lazyproperty(fn: Callable) -> Any:
    """Allows to avoid recomputing a property over and over.

    The result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All
    succeeding calls will use the value stored in the private property."""

    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def list_to_str(l: List[Text], delim: Text = ", ", quote: Text = "'") -> Text:
    return delim.join([quote + e + quote for e in l])


def ordered(obj: Any) -> Any:
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def module_path_from_object(o: Any) -> Text:
    """Returns the fully qualified class path of the instantiated object."""
    return o.__class__.__module__ + "." + o.__class__.__name__


def class_from_module_path(module_path: Text) -> Type[Any]:
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
        return globals()[module_path]


def json_to_string(obj: Any, **kwargs: Any) -> Text:
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def write_json_to_file(filename: Text, obj: Any, **kwargs: Any) -> None:
    """Write an object as a json string to a file."""

    write_to_file(filename, json_to_string(obj, **kwargs))


def write_to_file(filename: Text, text: Text) -> None:
    """Write a text to a file."""

    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(str(text))


def read_json_file(filename: Text) -> Any:
    """Read json from a file."""
    content = rasa.utils.io.read_file(filename)
    try:
        return simplejson.loads(content)
    except ValueError as e:
        raise ValueError(
            "Failed to read json from '{}'. Error: "
            "{}".format(os.path.abspath(filename), e)
        )


def build_entity(
    start: int, end: int, value: Text, entity_type: Text, **kwargs: Dict[Text, Any]
) -> Dict[Text, Any]:
    """Builds a standard entity dictionary.

    Adds additional keyword parameters."""

    entity = {"start": start, "end": end, "value": value, "entity": entity_type}

    entity.update(kwargs)
    return entity


def is_model_dir(model_dir: Text) -> bool:
    """Checks if the given directory contains a model and can be safely removed.

    specifically checks if the directory has no subdirectories and
    if all files have an appropriate ending."""
    allowed_extensions = {".json", ".pkl", ".dat"}
    dir_tree = list(os.walk(model_dir))
    if len(dir_tree) != 1:
        return False
    model_dir, child_dirs, files = dir_tree[0]
    file_extenstions = [os.path.splitext(f)[1] for f in files]
    only_valid_files = all([ext in allowed_extensions for ext in file_extenstions])
    return only_valid_files


def is_url(resource_name: Text) -> bool:
    """Return True if string is an http, ftp, or file URL path.

    This implementation is the same as the one used by matplotlib"""

    URL_REGEX = re.compile(r"http://|https://|ftp://|file://|file:\\")
    return URL_REGEX.match(resource_name) is not None


def remove_model(model_dir: Text) -> bool:
    """Removes a model directory and all its content."""
    import shutil

    if is_model_dir(model_dir):
        shutil.rmtree(model_dir)
        return True
    else:
        raise ValueError(
            "Cannot remove {}, it seems it is not a model "
            "directory".format(model_dir)
        )


def configure_colored_logging(loglevel: Text) -> None:
    import coloredlogs

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["debug"] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
        level_styles=level_styles,
        field_styles=field_styles,
    )


def pycloud_unpickle(file_name: Text) -> Any:
    """Unpickle an object from file using cloudpickle."""
    import cloudpickle

    with io.open(file_name, "rb") as f:  # pragma: no test
        return cloudpickle.load(f, encoding="latin-1")


def pycloud_pickle(file_name: Text, obj: Any) -> None:
    """Pickle an object to a file using cloudpickle."""
    import cloudpickle

    with io.open(file_name, "wb") as f:
        cloudpickle.dump(obj, f)


def create_temporary_file(data: Any, suffix: Text = "", mode: Text = "w+") -> Text:
    """Creates a tempfile.NamedTemporaryFile object for data.

    mode defines NamedTemporaryFile's  mode parameter in py3."""

    encoding = None if "b" in mode else "utf-8"
    f = tempfile.NamedTemporaryFile(
        mode=mode, suffix=suffix, delete=False, encoding=encoding
    )
    f.write(data)

    f.close()
    return f.name


def zip_folder(folder: Text) -> Text:
    """Create an archive from a folder."""
    import tempfile
    import shutil

    # WARN: not thread save!
    zipped_path = tempfile.NamedTemporaryFile(delete=False)
    zipped_path.close()
    return shutil.make_archive(zipped_path.name, str("zip"), folder)


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


def read_endpoints(endpoint_file: Text) -> "AvailableEndpoints":
    model = read_endpoint_config(endpoint_file, endpoint_type="model")
    data = read_endpoint_config(endpoint_file, endpoint_type="data")

    return AvailableEndpoints(model, data)


# The EndpointConfig class is currently used to define external endpoints
# for pulling NLU models from a server and training data
AvailableEndpoints = namedtuple("AvailableEndpoints", "model data")
