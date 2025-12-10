"""Tests for MCP project context resources."""

import json
from pathlib import Path

import pytest

from rasa.builder.copilot.mcp_server.resources.project_context import (
    _get_subfolder_files,
    get_assistant_logs,
    get_custom_actions_code,
    get_domain_definitions,
    get_flows_definitions,
)


class TestGetSubfolderFiles:
    """Test _get_subfolder_files helper function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with test files."""
        # Create data directory with flows
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "flows.yml").write_text("flows:\n  - greet_user")
        (data_dir / "nlu.yml").write_text("nlu:\n  - intent: greet")
        (data_dir / "readme.txt").write_text("not yaml")

        # Create domain directory
        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        (domain_dir / "slots.yml").write_text("slots:\n  name:\n    type: text")
        (domain_dir / "responses.yaml").write_text("responses:\n  utter_greet: []")

        # Create actions directory
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "actions.py").write_text("class MyAction:\n    pass")
        (actions_dir / "__init__.py").write_text("")
        (actions_dir / "config.yml").write_text("not python")

        return tmp_path

    def test_get_subfolder_files_data_directory(self, project_folder: Path) -> None:
        """Test getting files from data directory."""
        files = _get_subfolder_files(
            str(project_folder),
            subfolder="data",
            allowed_file_extensions=["yml", "yaml"],
        )

        assert "data/flows.yml" in files
        assert "data/nlu.yml" in files
        # txt files should be filtered out
        assert "data/readme.txt" not in files

    def test_get_subfolder_files_domain_directory(self, project_folder: Path) -> None:
        """Test getting files from domain directory."""
        files = _get_subfolder_files(
            str(project_folder),
            subfolder="domain",
            allowed_file_extensions=["yml", "yaml"],
        )

        assert "domain/slots.yml" in files
        assert "domain/responses.yaml" in files

    def test_get_subfolder_files_actions_directory(self, project_folder: Path) -> None:
        """Test getting files from actions directory."""
        files = _get_subfolder_files(
            str(project_folder),
            subfolder="actions",
            allowed_file_extensions=["py"],
        )

        assert "actions/actions.py" in files
        assert "actions/__init__.py" in files
        # yml files should be filtered out
        assert "actions/config.yml" not in files

    def test_get_subfolder_files_nonexistent_directory(
        self, project_folder: Path
    ) -> None:
        """Test getting files from nonexistent directory returns empty dict."""
        files = _get_subfolder_files(
            str(project_folder),
            subfolder="nonexistent",
            allowed_file_extensions=["yml"],
        )

        assert files == {}

    def test_get_subfolder_files_no_extension_filter(
        self, project_folder: Path
    ) -> None:
        """Test getting files without extension filter returns all files."""
        files = _get_subfolder_files(
            str(project_folder),
            subfolder="data",
            allowed_file_extensions=None,
        )

        assert "data/flows.yml" in files
        assert "data/nlu.yml" in files
        assert "data/readme.txt" in files


class TestGetFlowsDefinitions:
    """Test get_flows_definitions function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with flow files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "flows.yml").write_text("flows:\n  - greet_user")
        (data_dir / "more_flows.yaml").write_text("flows:\n  - farewell_user")
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_flows_definitions_returns_json(
        self, project_folder: Path
    ) -> None:
        """Test that get_flows_definitions returns valid JSON."""
        result = await get_flows_definitions(str(project_folder))

        parsed = json.loads(result)
        assert "flows" in parsed
        assert "count" in parsed

    @pytest.mark.asyncio
    async def test_get_flows_definitions_includes_yaml_files(
        self, project_folder: Path
    ) -> None:
        """Test that both .yml and .yaml files are included."""
        result = await get_flows_definitions(str(project_folder))

        parsed = json.loads(result)
        assert parsed["count"] == 2
        assert "data/flows.yml" in parsed["flows"]
        assert "data/more_flows.yaml" in parsed["flows"]

    @pytest.mark.asyncio
    async def test_get_flows_definitions_nonexistent_project(
        self, tmp_path: Path
    ) -> None:
        """Test get_flows_definitions with nonexistent project."""
        result = await get_flows_definitions(str(tmp_path / "nonexistent"))

        parsed = json.loads(result)
        assert parsed["count"] == 0


class TestGetDomainDefinitions:
    """Test get_domain_definitions function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with domain files."""
        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        (domain_dir / "slots.yml").write_text("slots:\n  name:\n    type: text")
        (domain_dir / "responses.yaml").write_text("responses:\n  utter_greet: []")
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_domain_definitions_returns_json(
        self, project_folder: Path
    ) -> None:
        """Test that get_domain_definitions returns valid JSON."""
        result = await get_domain_definitions(str(project_folder))

        parsed = json.loads(result)
        assert "domain" in parsed
        assert "count" in parsed

    @pytest.mark.asyncio
    async def test_get_domain_definitions_includes_files(
        self, project_folder: Path
    ) -> None:
        """Test that domain files are included."""
        result = await get_domain_definitions(str(project_folder))

        parsed = json.loads(result)
        assert parsed["count"] == 2
        assert "domain/slots.yml" in parsed["domain"]
        assert "domain/responses.yaml" in parsed["domain"]


class TestGetCustomActionsCode:
    """Test get_custom_actions_code function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with action files."""
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "actions.py").write_text(
            "from rasa_sdk import Action\n\nclass MyAction(Action):\n    pass"
        )
        (actions_dir / "__init__.py").write_text("# init")
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_custom_actions_code_returns_json(
        self, project_folder: Path
    ) -> None:
        """Test that get_custom_actions_code returns valid JSON."""
        result = await get_custom_actions_code(str(project_folder))

        parsed = json.loads(result)
        assert "actions" in parsed
        assert "count" in parsed

    @pytest.mark.asyncio
    async def test_get_custom_actions_code_includes_python_files(
        self, project_folder: Path
    ) -> None:
        """Test that Python files are included."""
        result = await get_custom_actions_code(str(project_folder))

        parsed = json.loads(result)
        assert parsed["count"] == 2
        assert "actions/actions.py" in parsed["actions"]
        assert "actions/__init__.py" in parsed["actions"]

    @pytest.mark.asyncio
    async def test_get_custom_actions_code_file_content(
        self, project_folder: Path
    ) -> None:
        """Test that file content is included."""
        result = await get_custom_actions_code(str(project_folder))

        parsed = json.loads(result)
        assert "class MyAction" in parsed["actions"]["actions/actions.py"]


class TestGetAssistantLogs:
    """Test get_assistant_logs function."""

    @pytest.mark.asyncio
    async def test_get_assistant_logs_returns_string(self) -> None:
        """Test that get_assistant_logs returns a string."""
        result = await get_assistant_logs()

        assert isinstance(result, str)
        # May be empty if no logs have been collected
