import builtins
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

import rasa.shared.utils.cli


def mock_print(*args, **kwargs):
    raise BlockingIOError()


def test_print_error_and_exit():
    with pytest.raises(SystemExit):
        rasa.shared.utils.cli.print_error_and_exit("")


def test_print_success(monkeypatch: MonkeyPatch):
    mock = Mock()
    monkeypatch.setattr(builtins, "print", mock_print)
    monkeypatch.setattr(rasa.shared.utils.io, "handle_print_blocking", mock)

    text = "Test Success"
    rasa.shared.utils.cli.print_success(text)

    assert mock.called
    assert (
        mock.call_args_list[0][0][0]
        == rasa.shared.utils.io.bcolors.OKGREEN
        + text
        + rasa.shared.utils.io.bcolors.ENDC
    )


def test_print_info(monkeypatch: MonkeyPatch):
    mock = Mock()
    monkeypatch.setattr(builtins, "print", mock_print)
    monkeypatch.setattr(rasa.shared.utils.io, "handle_print_blocking", mock)

    text = "Test Info"
    rasa.shared.utils.cli.print_info(text)

    assert mock.called
    assert (
        mock.call_args_list[0][0][0]
        == rasa.shared.utils.io.bcolors.OKBLUE
        + text
        + rasa.shared.utils.io.bcolors.ENDC
    )


def test_print_warning(monkeypatch: MonkeyPatch):
    mock = Mock()
    monkeypatch.setattr(builtins, "print", mock_print)
    monkeypatch.setattr(rasa.shared.utils.io, "handle_print_blocking", mock)

    text = "Test Warning"
    rasa.shared.utils.cli.print_warning(text)

    assert mock.called
    assert (
        mock.call_args_list[0][0][0]
        == rasa.shared.utils.io.bcolors.WARNING
        + text
        + rasa.shared.utils.io.bcolors.ENDC
    )


def test_print_error(monkeypatch: MonkeyPatch):
    mock = Mock()
    monkeypatch.setattr(builtins, "print", mock_print)
    monkeypatch.setattr(rasa.shared.utils.io, "handle_print_blocking", mock)

    text = "Test Error"
    rasa.shared.utils.cli.print_error(text)

    assert mock.called
    assert (
        mock.call_args_list[0][0][0]
        == rasa.shared.utils.io.bcolors.FAIL + text + rasa.shared.utils.io.bcolors.ENDC
    )
