import argparse
from typing import Callable, Text
from unittest.mock import Mock, ANY

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult

import rasa
from rasa.core.train import do_interactive_learning
from rasa.core.training import interactive as interactive_learning
from rasa.cli import interactive, train
from rasa.model_training import TrainingResult

from tests.cli.conftest import RASA_EXE


def test_interactive_help(run: Callable[..., RunResult]):
    output = run("interactive", "--help")

    help_text = f"""usage: {RASA_EXE} interactive [-h] [-v] [-vv] [--quiet]
                        [--logging-config-file LOGGING_CONFIG_FILE] [--e2e]
                        [-p PORT] [-m MODEL] [--data DATA [DATA ...]]
                        [--skip-visualization]
                        [--conversation-id CONVERSATION_ID]
                        [--endpoints ENDPOINTS] [-c CONFIG] [-d DOMAIN]
                        [--out OUT] [--augmentation AUGMENTATION]
                        [--debug-plots] [--finetune [FINETUNE]]
                        [--epoch-fraction EPOCH_FRACTION] [--force]
                        [--persist-nlu-data]
                        {{core}} ... [model-as-positional-argument]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_interactive_core_help(run: Callable[..., RunResult]):
    output = run("interactive", "core", "--help")

    help_text = f"""usage: {RASA_EXE} interactive core [-h] [-v] [-vv] [--quiet]
                             [--logging-config-file LOGGING_CONFIG_FILE]
                             [-m MODEL] [-s STORIES] [--skip-visualization]
                             [--conversation-id CONVERSATION_ID]
                             [--endpoints ENDPOINTS] [-c CONFIG] [-d DOMAIN]
                             [--out OUT] [--augmentation AUGMENTATION]
                             [--debug-plots] [--finetune [FINETUNE]]
                             [--epoch-fraction EPOCH_FRACTION] [-p PORT]
                             [model-as-positional-argument]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_pass_arguments_to_rasa_train(
    stack_config_path: Text, monkeypatch: MonkeyPatch
) -> None:
    # Create parser
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    # Parse interactive command
    args = parser.parse_args(["interactive", "--config", stack_config_path])
    interactive._set_not_required_args(args)

    # Mock actual training
    mock = Mock(return_value=TrainingResult(code=0))
    monkeypatch.setattr(rasa, "train", mock.method)

    # If the `Namespace` object does not have all required fields this will throw
    train.run_training(args)

    # Assert `train` was actually called
    mock.method.assert_called_once()


def test_train_called_when_no_model_passed(
    stack_config_path: Text, monkeypatch: MonkeyPatch
) -> None:
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    args = parser.parse_args(
        [
            "interactive",
            "--config",
            stack_config_path,
            "--data",
            "data/test_moodbot/data",
        ]
    )
    interactive._set_not_required_args(args)

    # Mock actual training and interactive learning methods
    mock = Mock()
    monkeypatch.setattr(train, "run_training", mock.train_model)
    monkeypatch.setattr(
        interactive, "perform_interactive_learning", mock.perform_interactive_learning
    )

    interactive.interactive(args)
    mock.train_model.assert_called_once()


def test_train_core_called_when_no_model_passed_and_core(
    stack_config_path: Text, monkeypatch: MonkeyPatch
) -> None:
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    args = parser.parse_args(
        [
            "interactive",
            "core",
            "--config",
            stack_config_path,
            "--stories",
            "data/test_moodbot/data/stories.yml",
            "--domain",
            "data/test_moodbot/domain.yml",
        ]
    )
    interactive._set_not_required_args(args)

    # Mock actual training and interactive learning methods
    mock = Mock()
    monkeypatch.setattr(train, "run_core_training", mock.run_core_training)
    monkeypatch.setattr(
        interactive, "perform_interactive_learning", mock.perform_interactive_learning
    )

    interactive.interactive(args)
    mock.run_core_training.assert_called_once()


def test_no_interactive_without_core_data(
    stack_config_path: Text, monkeypatch: MonkeyPatch, nlu_data_path: Text
) -> None:
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    args = parser.parse_args(
        ["interactive", "--config", stack_config_path, "--data", nlu_data_path]
    )
    interactive._set_not_required_args(args)

    mock = Mock()
    monkeypatch.setattr(train, "run_training", mock.train_model)
    monkeypatch.setattr(
        interactive, "perform_interactive_learning", mock.perform_interactive_learning
    )

    with pytest.raises(SystemExit):
        interactive.interactive(args)

    mock.train_model.assert_not_called()
    mock.perform_interactive_learning.assert_not_called()


def test_pass_conversation_id_to_interactive_learning(monkeypatch: MonkeyPatch):
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

    _serve_application.assert_called_once_with(
        ANY, ANY, True, expected_conversation_id, 5005
    )


def test_generate_conversation_id_for_interactive_learning(monkeypatch: MonkeyPatch):
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()
    interactive.add_subparser(sub_parser, [])

    args = parser.parse_args(["interactive"])

    assert args.conversation_id
