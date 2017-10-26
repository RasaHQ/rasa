from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import errno
import os, io
from collections import deque
from hashlib import sha1

import six
import yaml
from builtins import input, range, str
from numpy import all, array
from typing import Text


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


def all_subclasses(cls):
    """Returns all known (imported) subclasses of a class."""

    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


def subsample_array(arr, max_values, can_modify_incoming_array=True, rand=None):
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


class TopicStack(object):
    def __init__(self, topics, iterable, default):
        self.topics = topics
        self.iterable = iterable
        self.topic_names = [t.name for t in topics]
        self.default = default
        self.dq = deque(iterable, len(topics))

    @property
    def top(self):
        if len(self.dq) < 1:
            return self.default
        return self.dq[-1]

    def __iter__(self):
        return self.dq.__iter__()

    def next(self):
        return self.dq.next()

    def __len__(self):
        return len(self.dq)

    def push(self, x):
        from rasa_core.conversation import Topic

        if isinstance(x, six.string_types):
            if x not in self.topic_names:
                raise ValueError(
                        "Unknown topic name: '{}', known topics in this domain "
                        "are: {}".format(x, self.topic_names))
            else:
                x = self.topics[self.topic_names.index(x)]

        elif not isinstance(x, Topic) or x not in self.topics:
            raise ValueError(
                    "Instance of type '{}' can not be used on the topic stack, "
                    "not a valid topic!".format(type(x).__name__))

        while self.dq.count(x) > 0:
            self.dq.remove(x)
        self.dq.append(x)

    def pop(self):
        if len(self.dq) < 1:
            return None
        return self.dq.pop()


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
    from yaml import Loader, SafeLoader

    def construct_yaml_str(self, node):
        # Override the default string handling function
        # to always return unicode objects
        return self.construct_scalar(node)

    Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)
    SafeLoader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)


def read_yaml_file(filename):
    fix_yaml_loader()
    with io.open(filename, encoding="utf-8") as f:
        return yaml.load(f.read())


def is_training_data_empty(X):
    """Check if the training matrix does contain training samples."""
    return X.shape[0] == 0
