import argparse
from typing import Callable, Text
from unittest.mock import Mock

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult

import rasa
from rasa.cli import interactive, train


def test_interactive_help(run: Callable[..., RunResult]):
    output = run("interactive", "--help")

    help_text = """usage: rasa interactive [-h] [-v] [-vv] [--quiet] [--e2e] [-m MODEL]
                        [--data DATA [DATA ...]] [--skip-visualization]
                        [--endpoints ENDPOINTS] [-c CONFIG] [-d DOMAIN]
                        [--out OUT] [--augmentation AUGMENTATION]
                        [--debug-plots] [--dump-stories] [--force]
                        [--persist-nlu-data]
                        {core} ... [model-as-positional-argument]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_interactive_core_help(run: Callable[..., RunResult]):
    output = run("interactive", "core", "--help")

    help_text = """usage: rasa interactive core [-h] [-v] [-vv] [--quiet] [-m MODEL] [-s STORIES]
                             [--skip-visualization] [--endpoints ENDPOINTS]
                             [-c CONFIG] [-d DOMAIN] [--out OUT]
                             [--augmentation AUGMENTATION] [--debug-plots]
                             [--dump-stories]
                             [model-as-positional-argument]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_pass_arguments_to_rasa_train(
    default_stack_config: Text, monkeypatch: MonkeyPatch
) -> None:
    # Create parser
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    # Parse interactive command
    args = parser.parse_args(["interactive", "--config", default_stack_config])
    interactive._set_not_required_args(args)

    # Mock actual training
    mock = Mock()
    monkeypatch.setattr(rasa, "train", mock.method)

    # If the `Namespace` object does not have all required fields this will throw
    train.train(args)

    # Assert `train` was actually code
    mock.method.assert_called_once()
