import argparse
import pytest
from typing import Callable, Text
from unittest.mock import Mock, ANY

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult

import rasa
from rasa.cli import interactive, train


def test_interactive_help(run: Callable[..., RunResult]):
    output = run("interactive", "--help")

    help_text = """usage: rasa interactive [-h] [-v] [-vv] [--quiet] [--e2e] [-m MODEL]
                        [--data DATA [DATA ...]] [--skip-visualization]
                        [--conversation-id CONVERSATION_ID]
                        [--endpoints ENDPOINTS] [-c CONFIG] [-d DOMAIN]
                        [--out OUT] [--augmentation AUGMENTATION]
                        [--debug-plots] [--force] [--persist-nlu-data]
                        {core} ... [model-as-positional-argument]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_interactive_core_help(run: Callable[..., RunResult]):
    output = run("interactive", "core", "--help")

    help_text = """usage: rasa interactive core [-h] [-v] [-vv] [--quiet] [-m MODEL] [-s STORIES]
                             [--skip-visualization]
                             [--conversation-id CONVERSATION_ID]
                             [--endpoints ENDPOINTS] [-c CONFIG] [-d DOMAIN]
                             [--out OUT] [--augmentation AUGMENTATION]
                             [--debug-plots]
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

    # Assert `train` was actually called
    mock.method.assert_called_once()


def test_train_called_when_no_model_passed(
    default_stack_config: Text, monkeypatch: MonkeyPatch
) -> None:
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    args = parser.parse_args(
        [
            "interactive",
            "--config",
            default_stack_config,
            "--data",
            "examples/moodbot/data",
        ]
    )
    interactive._set_not_required_args(args)

    # Mock actual training and interactive learning methods
    mock = Mock()
    monkeypatch.setattr(train, "train", mock.train_model)
    monkeypatch.setattr(
        interactive, "perform_interactive_learning", mock.perform_interactive_learning
    )

    interactive.interactive(args)
    mock.train_model.assert_called_once()


def test_train_core_called_when_no_model_passed_and_core(
    default_stack_config: Text, monkeypatch: MonkeyPatch
) -> None:
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    args = parser.parse_args(
        [
            "interactive",
            "core",
            "--config",
            default_stack_config,
            "--stories",
            "examples/moodbot/data/stories.md",
            "--domain",
            "examples/moodbot/domain.yml",
        ]
    )
    interactive._set_not_required_args(args)

    # Mock actual training and interactive learning methods
    mock = Mock()
    monkeypatch.setattr(train, "train_core", mock.train_core)
    monkeypatch.setattr(
        interactive, "perform_interactive_learning", mock.perform_interactive_learning
    )

    interactive.interactive(args)
    mock.train_core.assert_called_once()


def test_no_interactive_without_core_data(
    default_stack_config: Text, monkeypatch: MonkeyPatch
) -> None:
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    args = parser.parse_args(
        [
            "interactive",
            "--config",
            default_stack_config,
            "--data",
            "examples/moodbot/data/nlu.md",
        ]
    )
    interactive._set_not_required_args(args)

    mock = Mock()
    monkeypatch.setattr(train, "train", mock.train_model)
    monkeypatch.setattr(
        interactive, "perform_interactive_learning", mock.perform_interactive_learning
    )

    with pytest.raises(SystemExit):
        interactive.interactive(args)

    mock.train_model.assert_not_called()
    mock.perform_interactive_learning.assert_not_called()


def test_pass_conversation_id_to_interactive_learning(monkeypatch: MonkeyPatch):
    from rasa.core.train import do_interactive_learning
    from rasa.core.training import interactive as interactive_learning

    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    expected_conversation_id = "🎁"
    args = parser.parse_args(
        [
            "interactive",
            "--conversation-id",
            expected_conversation_id,
            "--skip-visualization",
        ]
    )

    _serve_application = Mock()
    monkeypatch.setattr(interactive_learning, "_serve_application", _serve_application)

    do_interactive_learning(args, Mock())

    _serve_application.assert_called_once_with(ANY, ANY, True, expected_conversation_id)


def test_generate_conversation_id_for_interactive_learning(monkeypatch: MonkeyPatch):
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    args = parser.parse_args(["interactive"])

    assert args.conversation_id
