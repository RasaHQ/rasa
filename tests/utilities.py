from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib
import io
import itertools
import os
import sys

import jsonpickle
import six

from rasa_core.domain import TemplateDomain
from rasa_core.trackers import DialogueStateTracker
from tests.conftest import DEFAULT_DOMAIN_PATH


def tracker_from_dialogue_file(filename, domain=None):
    dialogue = read_dialogue_file(filename)

    if domain is not None:
        domain = domain
    else:
        domain = TemplateDomain.load(DEFAULT_DOMAIN_PATH)
    tracker = DialogueStateTracker(dialogue.name, domain.slots)
    tracker.recreate_from_dialogue(dialogue)
    return tracker


def read_dialogue_file(filename):
    with io.open(filename, "r") as f:
        dialogue_json = f.read()
    return jsonpickle.loads(dialogue_json)


def write_text_to_file(tmpdir, filename, text):
    path = tmpdir.join(filename).strpath
    with io.open(path, "w") as f:
        f.write(text)
    return path


@contextlib.contextmanager
def cwd(path):
    CWD = os.getcwd()

    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(CWD)


@contextlib.contextmanager
def mocked_cmd_input(package, text):
    if isinstance(text, six.string_types):
        text = [text]

    text_generator = itertools.cycle(text)
    i = package.input

    def mocked_input(_=None):
        value = text_generator.next()
        print("wrote '{}' to input".format(value))
        return value

    package.input = mocked_input
    try:
        yield
    finally:
        package.input = i
