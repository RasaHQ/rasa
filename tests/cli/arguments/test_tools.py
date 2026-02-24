import argparse
from pathlib import Path
from typing import List, Optional

import pytest

from rasa.cli.tools import (
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
    RunConfig,
    _validate_config_exclusivity,
)
from rasa.shared.exceptions import RasaException


def test_default_tools_run_arguments(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests default settings for `rasa tools run` CLI command.

    Mode and port default to None at the argparse level; actual defaults are
    applied by the config resolution layer (``resolve_run_config``).
    """
    args = tools_parser.parse_args(["tools", "run"])

    assert args.project is None
    assert args.mode is None
    assert args.port is None
    assert args.config is None


def test_tools_run_with_mode_stdio(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --mode stdio`."""
    args = tools_parser.parse_args(["tools", "run", "--mode", "stdio"])

    assert args.mode == "stdio"


def test_tools_run_with_mode_http(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --mode http`."""
    args = tools_parser.parse_args(["tools", "run", "--mode", "http"])

    assert args.mode == "http"


def test_tools_run_with_invalid_mode(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests that invalid --mode value is rejected."""
    with pytest.raises(SystemExit):
        tools_parser.parse_args(["tools", "run", "--mode", "grpc"])


def test_tools_run_with_port(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --port 9999`."""
    args = tools_parser.parse_args(["tools", "run", "--port", "9999"])

    assert args.port == 9999


def test_tools_run_with_invalid_port(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests that non-integer --port value is rejected."""
    with pytest.raises(SystemExit):
        tools_parser.parse_args(["tools", "run", "--port", "not-a-number"])


def test_tools_run_with_project(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --project /some/path`."""
    args = tools_parser.parse_args(["tools", "run", "--project", "/some/path"])

    assert args.project == "/some/path"


def test_tools_run_with_config(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --config /some/config.yaml`."""
    args = tools_parser.parse_args(["tools", "run", "--config", "/some/config.yaml"])

    assert args.config == "/some/config.yaml"


@pytest.mark.parametrize(
    "input_args, expected_mode, expected_port, expected_project",
    [
        (
            [
                "tools",
                "run",
                "--mode",
                "http",
                "--port",
                "8080",
                "--project",
                "/my/bot",
            ],
            "http",
            8080,
            "/my/bot",
        ),
        (
            ["tools", "run", "--mode", "stdio", "--project", "."],
            "stdio",
            None,
            ".",
        ),
        (
            ["tools", "run", "--port", "7777"],
            None,
            7777,
            None,
        ),
    ],
)
def test_tools_run_combined_arguments(
    tools_parser: argparse.ArgumentParser,
    input_args: List[str],
    expected_mode: Optional[str],
    expected_port: Optional[int],
    expected_project: Optional[str],
) -> None:
    """Tests various combinations of `rasa tools run` arguments."""
    args = tools_parser.parse_args(input_args)

    assert args.mode == expected_mode
    assert args.port == expected_port
    assert args.project == expected_project


def test_config_alone_is_valid() -> None:
    args = argparse.Namespace(
        config="/some/config.yaml", mode=None, port=None, project=None
    )
    _validate_config_exclusivity(args)


def test_no_config_allows_other_args() -> None:
    args = argparse.Namespace(config=None, mode="http", port=8080, project="/p")
    _validate_config_exclusivity(args)


@pytest.mark.parametrize(
    "mode, port, project",
    [
        ("http", None, None),
        (None, 8080, None),
        (None, None, "/p"),
        ("http", 8080, None),
        ("http", None, "/p"),
        (None, 8080, "/p"),
        ("http", 8080, "/p"),
    ],
)
def test_config_with_other_args_raises(
    mode: Optional[str],
    port: Optional[int],
    project: Optional[str],
) -> None:
    args = argparse.Namespace(
        config="/some/config.yaml", mode=mode, port=port, project=project
    )
    with pytest.raises(RasaException, match="--config cannot be combined"):
        _validate_config_exclusivity(args)


class TestRunConfigPersistence:
    def test_save_creates_directory_and_file(self, tmp_path: Path) -> None:
        cfg = RunConfig(mode="http", port=1234)
        dest = tmp_path / "nested" / "dir" / "tools.yaml"
        cfg.save(dest)

        assert dest.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = RunConfig(mode="http", port=4567)
        dest = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        cfg.save(dest)

        loaded = RunConfig.load(dest)
        assert loaded.mode == "http"
        assert loaded.port == 4567

    def test_load_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(RasaException, match="does not exist"):
            RunConfig.load(tmp_path / "nonexistent.yaml")

    def test_load_raises_for_non_dict_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(RasaException, match="Invalid config"):
            RunConfig.load(bad)

    def test_load_handles_extra_keys_gracefully(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "tools.yaml"
        cfg_file.write_text("mode: http\nport: 5000\nfuture_key: value\n")
        loaded = RunConfig.load(cfg_file)
        assert loaded.mode == "http"
        assert loaded.port == 5000
