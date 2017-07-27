from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

import errno
from typing import List
from typing import Optional
from typing import Text


def relative_normpath(f, path):
    # type: (Optional[Text], Text) -> Optional[Text]
    """Return the path of file relative to `path`."""

    if f is not None:
        return os.path.normpath(os.path.relpath(f, path))
    else:
        return None


def create_dir(dir_path):
    # type: (Text) -> None
    """Creates a directory and its super paths. Succeeds even if the path already exists."""

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


def recursively_find_files(resource_name):
    # type: (Optional[Text]) -> List[Text]
    """Traverse directory hierarchy to find files.

    `resource_name` can be a folder or a file. In both cases we will return a list of files."""

    if not resource_name:
        raise ValueError("Resource name '{}' must be an existing directory or file.".format(resource_name))
    elif os.path.isfile(resource_name):
        return [resource_name]
    elif os.path.isdir(resource_name):
        resources = []  # type: List[Text]
        # walk the fs tree and return a list of files
        nodes_to_visit = [resource_name]
        while len(nodes_to_visit) > 0:
            # skip hidden files
            nodes_to_visit = [f for f in nodes_to_visit if not f.split("/")[-1].startswith('.')]

            current_node = nodes_to_visit[0]
            # if current node is a folder, schedule its children for a visit. Else add them to the resources.
            if os.path.isdir(current_node):
                nodes_to_visit += [os.path.join(current_node, f) for f in os.listdir(current_node)]
            else:
                resources += [current_node]
            nodes_to_visit = nodes_to_visit[1:]
        return resources
    else:
        raise ValueError("Could not locate the resource '{}'.".format(os.path.abspath(resource_name)))


def lazyproperty(fn):
    """Allows to avoid recomputing a property over and over. Instead the result gets stored in a local var.

    Computation of the property will happen once, on the first call of the property. All succeeding calls will use
    the value stored in the private property."""

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
