import argparse
from typing import List, Optional

import pytest

from rasa.cli.tools.run import _validate_config_exclusivity
from rasa.shared.exceptions import RasaException


class TestToolsInitArguments:
    def test_default_init_arguments(
        self, tools_parser: argparse.ArgumentParser
    ) -> None:
        args = tools_parser.parse_args(["tools", "init"])
        assert args.yes is False
        assert args.project_path is None
        assert args.mode is None
        assert args.port is None
        assert args.docs is None
        assert args.ides is None
        assert args.rasa_server_url is None

    def test_init_yes_flag(self, tools_parser: argparse.ArgumentParser) -> None:
        args = tools_parser.parse_args(["tools", "init", "--yes"])
        assert args.yes is True

    def test_init_yes_short_flag(self, tools_parser: argparse.ArgumentParser) -> None:
        args = tools_parser.parse_args(["tools", "init", "-y"])
        assert args.yes is True

    def test_init_all_flags(self, tools_parser: argparse.ArgumentParser) -> None:
        args = tools_parser.parse_args(
            [
                "tools",
                "init",
                "--yes",
                "--project-path",
                "/my/bot",
                "--mode",
                "http",
                "--port",
                "9000",
                "--docs",
                "online",
                "--ides",
                "cursor,vscode",
                "--rasa-server-url",
                "http://my-server:9999",
            ]
        )
        assert args.yes is True
        assert args.project_path == "/my/bot"
        assert args.mode == "http"
        assert args.port == 9000
        assert args.docs == "online"
        assert args.ides == "cursor,vscode"
        assert args.rasa_server_url == "http://my-server:9999"

    def test_init_invalid_mode_rejected(
        self, tools_parser: argparse.ArgumentParser
    ) -> None:
        with pytest.raises(SystemExit):
            tools_parser.parse_args(["tools", "init", "--mode", "grpc"])

    def test_init_invalid_docs_rejected(
        self, tools_parser: argparse.ArgumentParser
    ) -> None:
        with pytest.raises(SystemExit):
            tools_parser.parse_args(["tools", "init", "--docs", "hybrid"])


def test_default_tools_run_arguments(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests default settings for `rasa tools run` CLI command.

    Mode and port default to None at the argparse level; actual defaults are
    applied by the config resolution layer (``resolve_run_config``).
    """
    args = tools_parser.parse_args(["tools", "run"])

    assert args.project_path is None
    assert args.mode is None
    assert args.port is None
    assert args.config is None
    assert args.rasa_server_url is None


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
    """Tests `rasa tools run --project-path /some/path`."""
    args = tools_parser.parse_args(["tools", "run", "--project-path", "/some/path"])

    assert args.project_path == "/some/path"


def test_tools_run_with_rasa_server_url(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --rasa-server-url http://localhost:5005`."""
    args = tools_parser.parse_args(
        ["tools", "run", "--rasa-server-url", "http://localhost:5005"]
    )

    assert args.rasa_server_url == "http://localhost:5005"


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
                "--project-path",
                "/my/bot",
            ],
            "http",
            8080,
            "/my/bot",
        ),
        (
            ["tools", "run", "--mode", "stdio", "--project-path", "."],
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
    assert args.project_path == expected_project


def test_config_alone_is_valid() -> None:
    args = argparse.Namespace(
        config="/some/config.yaml",
        mode=None,
        port=None,
        project_path=None,
        rasa_server_url=None,
    )
    _validate_config_exclusivity(args)


def test_no_config_allows_other_args() -> None:
    args = argparse.Namespace(
        config=None,
        mode="http",
        port=8080,
        project_path="/p",
        rasa_server_url=None,
    )
    _validate_config_exclusivity(args)


@pytest.mark.parametrize(
    "mode, port, project_path, rasa_server_url",
    [
        ("http", None, None, None),
        (None, 8080, None, None),
        (None, None, "/p", None),
        ("http", 8080, None, None),
        ("http", None, "/p", None),
        (None, 8080, "/p", None),
        ("http", 8080, "/p", None),
        (None, None, None, "http://localhost:5005"),
    ],
)
def test_config_with_other_args_raises(
    mode: Optional[str],
    port: Optional[int],
    project_path: Optional[str],
    rasa_server_url: Optional[str],
) -> None:
    args = argparse.Namespace(
        config="/some/config.yaml",
        mode=mode,
        port=port,
        project_path=project_path,
        rasa_server_url=rasa_server_url,
    )
    with pytest.raises(RasaException, match="--config cannot be combined"):
        _validate_config_exclusivity(args)


class TestToolsInitSkillsArguments:
    def test_default_skills_arguments(
        self, tools_parser: argparse.ArgumentParser
    ) -> None:
        args = tools_parser.parse_args(["tools", "init", "skills"])
        assert args.project_path is None
        assert args.ides is None
        assert args.yes is False

    def test_skills_yes_flag(self, tools_parser: argparse.ArgumentParser) -> None:
        args = tools_parser.parse_args(["tools", "init", "skills", "--yes"])
        assert args.yes is True

    def test_skills_yes_short_flag(self, tools_parser: argparse.ArgumentParser) -> None:
        args = tools_parser.parse_args(["tools", "init", "skills", "-y"])
        assert args.yes is True

    def test_skills_with_project_path(
        self, tools_parser: argparse.ArgumentParser
    ) -> None:
        args = tools_parser.parse_args(
            ["tools", "init", "skills", "--project-path", "/my/bot"]
        )
        assert args.project_path == "/my/bot"

    def test_skills_with_ides(self, tools_parser: argparse.ArgumentParser) -> None:
        args = tools_parser.parse_args(
            ["tools", "init", "skills", "--ides", "cursor,vscode"]
        )
        assert args.ides == "cursor,vscode"

    def test_skills_with_all_flags(self, tools_parser: argparse.ArgumentParser) -> None:
        args = tools_parser.parse_args(
            [
                "tools",
                "init",
                "skills",
                "-y",
                "--project-path",
                "/my/bot",
                "--ides",
                "cursor,vscode,claude",
            ]
        )
        assert args.yes is True
        assert args.project_path == "/my/bot"
        assert args.ides == "cursor,vscode,claude"


class TestToolsInitDocsArguments:
    def test_default_docs_arguments(
        self, tools_parser: argparse.ArgumentParser
    ) -> None:
        args = tools_parser.parse_args(["tools", "init", "docs"])
        assert args.project_path is None

    def test_docs_with_project_path(
        self, tools_parser: argparse.ArgumentParser
    ) -> None:
        args = tools_parser.parse_args(
            ["tools", "init", "docs", "--project-path", "/my/bot"]
        )
        assert args.project_path == "/my/bot"
