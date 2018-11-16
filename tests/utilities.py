import itertools

import contextlib
import io
import jsonpickle
import os

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.trackers import DialogueStateTracker
from tests.conftest import DEFAULT_DOMAIN_PATH


def tracker_from_dialogue_file(filename, domain=None):
    dialogue = read_dialogue_file(filename)

    if domain is not None:
        domain = domain
    else:
        domain = Domain.load(DEFAULT_DOMAIN_PATH)
    tracker = DialogueStateTracker(dialogue.name, domain.slots)
    tracker.recreate_from_dialogue(dialogue)
    return tracker


def read_dialogue_file(filename):
    return jsonpickle.loads(utils.read_file(filename))


def write_text_to_file(tmpdir, filename, text):
    path = tmpdir.join(filename).strpath
    with io.open(path, "w", encoding="utf-8") as f:
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
    if isinstance(text, str):
        text = [text]

    text_generator = itertools.cycle(text)
    i = package.input

    def mocked_input(_=None):
        value = next(text_generator)
        print("wrote '{}' to input".format(value))
        return value

    package.input = mocked_input
    try:
        yield
    finally:
        package.input = i
