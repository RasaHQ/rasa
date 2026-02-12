"""Tests for MCP project context tools."""

from pathlib import Path

import pytest

from rasa.builder.copilot.mcp_server.tools.project_context import (
    _get_domain_yaml_files,
    _get_flow_yaml_files,
    _glob_yaml_files,
    get_assistant_logs,
    get_project_flow,
    get_project_response,
    get_project_slot,
    list_project_custom_actions,
    list_project_flows,
    list_project_responses,
    list_project_slots,
)
from rasa.utils.io import InvalidPathException


@pytest.fixture
def rasa_project(tmp_path: Path) -> Path:
    """Create a complete Rasa project with flows, slots, responses, and actions.

    This fixture creates files that contain multiple definition types to verify
    that each tool correctly extracts only its relevant definitions.
    """
    # Create data directory with flows
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "flows.yml").write_text(
        "flows:\n"
        "  greet_user:\n"
        "    name: Greet User\n"
        "    steps:\n"
        "      - action: utter_greet\n"
    )
    (data_dir / "more_flows.yaml").write_text(
        "flows:\n"
        "  farewell_user:\n"
        "    name: Farewell User\n"
        "    steps:\n"
        "      - action: utter_goodbye\n"
    )

    # Create subdirectory with additional flows
    flows_subdir = data_dir / "flows"
    flows_subdir.mkdir()
    (flows_subdir / "booking_flows.yml").write_text(
        "flows:\n"
        "  book_appointment:\n"
        "    name: Book Appointment\n"
        "    steps:\n"
        "      - action: action_book\n"
    )

    # Create domain directory with slots, responses, and actions in same file
    # This tests that each tool correctly ignores other definitions
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()
    (domain_dir / "domain.yml").write_text(
        "slots:\n"
        "  name:\n"
        "    type: text\n"
        "  age:\n"
        "    type: float\n"
        "\n"
        "responses:\n"
        "  utter_greet:\n"
        "    - text: Hello!\n"
        "  utter_goodbye:\n"
        "    - text: Goodbye!\n"
        "\n"
        "actions:\n"
        "  - action_search\n"
        "  - action_submit\n"
        "  - action_with_config:\n"  # Action as dict
        "      some_config: value\n"
        "  - utter_greet\n"  # Should be filtered out (response prefix)
    )

    # Create subdirectory with additional domain definitions
    domain_subdir = domain_dir / "subdir"
    domain_subdir.mkdir()
    (domain_subdir / "user_slots.yml").write_text(
        "slots:\n" "  email:\n" "    type: text\n"
    )
    (domain_subdir / "user_responses.yml").write_text(
        "responses:\n" "  utter_ask_email:\n" "    - text: What is your email?\n"
    )
    (domain_subdir / "user_actions.yml").write_text(
        "actions:\n" "  - action_validate_email\n"
    )

    return tmp_path


class TestGlobYamlFiles:
    """Test _glob_yaml_files helper function."""

    @pytest.fixture
    def yaml_dir(self, tmp_path: Path) -> Path:
        """Create a directory with various YAML files including subdirectories."""
        # Use different base names to avoid case-insensitive filesystem issues (macOS)
        (tmp_path / "file1.yml").write_text("key: value")
        (tmp_path / "file2.yaml").write_text("key: value")
        (tmp_path / "file3.YML").write_text("key: value")
        (tmp_path / "file4.YAML").write_text("key: value")
        (tmp_path / "file5.txt").write_text("not yaml")

        # Create subdirectory with YAML files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file6.yml").write_text("key: value")
        (subdir / "file7.yaml").write_text("key: value")

        # Create nested subdirectory
        nested_subdir = subdir / "nested"
        nested_subdir.mkdir()
        (nested_subdir / "file8.yml").write_text("key: value")

        return tmp_path

    def test_glob_yaml_files_finds_all_yaml_extensions_and_subdirectories(
        self, yaml_dir: Path
    ) -> None:
        """Test that all YAML extensions are found in directory and subdirectories."""
        files = list(_glob_yaml_files(yaml_dir))
        file_names = [f.name.lower() for f in files]

        # Check all yaml files are found (case-insensitive comparison)
        assert "file1.yml" in file_names
        assert "file2.yaml" in file_names
        assert "file3.yml" in file_names
        assert "file4.yaml" in file_names
        assert "file6.yml" in file_names  # From subdirectory
        assert "file7.yaml" in file_names  # From subdirectory
        assert "file8.yml" in file_names  # From nested subdirectory
        assert "file5.txt" not in file_names
        assert len(files) == 7


class TestGetDomainYamlFiles:
    """Test _get_domain_yaml_files helper function."""

    @pytest.fixture
    def project_with_domain_file(self, tmp_path: Path) -> Path:
        """Create project with single domain.yml file."""
        (tmp_path / "domain.yml").write_text("slots:\n  name:\n    type: text")
        return tmp_path

    def test_get_domain_yaml_files_from_directory_and_subdirectories(
        self, rasa_project: Path
    ) -> None:
        """Test getting domain files from domain directory and all subdirectories."""
        files = list(_get_domain_yaml_files(rasa_project, "domain"))
        file_names = [f.name for f in files]

        # Should find files in domain/ and all subdirectories
        assert "domain.yml" in file_names
        assert "user_slots.yml" in file_names
        assert "user_responses.yml" in file_names
        assert "user_actions.yml" in file_names
        assert len(files) == 4

    def test_get_domain_yaml_files_fallback_to_domain_yml(
        self, project_with_domain_file: Path
    ) -> None:
        """Test fallback to domain.yml when directory doesn't exist."""
        files = list(_get_domain_yaml_files(project_with_domain_file, "domain"))

        assert len(files) == 1
        assert files[0].name == "domain.yml"

    def test_get_domain_yaml_files_none_uses_domain_yml(
        self, project_with_domain_file: Path
    ) -> None:
        """Test that None domain_folder uses domain.yml directly."""
        files = list(_get_domain_yaml_files(project_with_domain_file, None))

        assert len(files) == 1
        assert files[0].name == "domain.yml"

    def test_get_domain_yaml_files_empty_project(self, tmp_path: Path) -> None:
        """Test with project that has no domain files."""
        files = list(_get_domain_yaml_files(tmp_path, "domain"))

        assert len(files) == 0


class TestGetFlowYamlFiles:
    """Test _get_flow_yaml_files helper function."""

    def test_get_flow_yaml_files_from_data_directory_and_subdirectories(
        self, rasa_project: Path
    ) -> None:
        """Test getting flow files from data directory and all subdirectories."""
        files = list(_get_flow_yaml_files(rasa_project, "data"))
        file_names = [f.name for f in files]

        # Should find files in data/ and all subdirectories
        assert "flows.yml" in file_names
        assert "more_flows.yaml" in file_names
        assert "booking_flows.yml" in file_names
        assert len(files) == 3

    def test_get_flow_yaml_files_none_searches_entire_project(
        self, rasa_project: Path
    ) -> None:
        """Test that None data_folder searches entire project."""
        files = list(_get_flow_yaml_files(rasa_project, None))
        file_names = [f.name for f in files]

        assert "flows.yml" in file_names


class TestListProjectFlows:
    """Test list_project_flows function."""

    @pytest.mark.asyncio
    async def test_list_project_flows(self, rasa_project: Path) -> None:
        """Test listing flows from a Rasa project including subdirectories."""
        # Given: A Rasa project with flow definitions in data folder and subdirectories

        # When: Listing all flows
        result = await list_project_flows(str(rasa_project), "data")

        # Then: Returns successful response with flow metadata from all subdirectories
        assert result.success is True
        assert result.error is None
        assert result.count == 3

        flow_ids = [f.id for f in result.flows]
        assert "greet_user" in flow_ids
        assert "farewell_user" in flow_ids
        assert "book_appointment" in flow_ids  # From subdirectory

        greet_flow = next(f for f in result.flows if f.id == "greet_user")
        assert greet_flow.name == "Greet User"
        assert "flows.yml" in greet_flow.file_path

        # Verify subdirectory flow has correct path
        book_flow = next(f for f in result.flows if f.id == "book_appointment")
        assert book_flow.name == "Book Appointment"
        assert "flows/booking_flows.yml" in book_flow.file_path

    @pytest.mark.asyncio
    async def test_list_project_flows_empty_project(self, tmp_path: Path) -> None:
        """Test listing flows from an empty project."""
        # Given: A project with no flow files

        # When: Listing flows
        result = await list_project_flows(str(tmp_path / "nonexistent"), "data")

        # Then: Returns success with empty list
        assert result.success is True
        assert result.count == 0


class TestListProjectSlots:
    """Test list_project_slots function."""

    @pytest.mark.asyncio
    async def test_list_project_slots(self, rasa_project: Path) -> None:
        """Test listing slots from domain including subdirectories."""
        # Given: Domain files in main directory and subdirectories

        # When: Listing all slots
        result = await list_project_slots(str(rasa_project), "domain")

        # Then: Returns slot definitions from all subdirectories
        assert result.success is True
        assert result.error is None
        assert result.count == 3

        slot_names = [s.name for s in result.slots]
        assert "name" in slot_names
        assert "age" in slot_names
        assert "email" in slot_names  # From subdirectory

        name_slot = next(s for s in result.slots if s.name == "name")
        assert name_slot.type == "text"
        assert "domain.yml" in name_slot.file_path

        # Verify subdirectory slot has correct path
        email_slot = next(s for s in result.slots if s.name == "email")
        assert email_slot.type == "text"
        assert "subdir/user_slots.yml" in email_slot.file_path


class TestListProjectResponses:
    """Test list_project_responses function."""

    @pytest.mark.asyncio
    async def test_list_project_responses(self, rasa_project: Path) -> None:
        """Test listing responses from domain including subdirectories."""
        # Given: Domain files in main directory and subdirectories

        # When: Listing all responses
        result = await list_project_responses(str(rasa_project), "domain")

        # Then: Returns response definitions from all subdirectories
        assert result.success is True
        assert result.error is None
        assert result.count == 3

        response_names = [r.name for r in result.responses]
        assert "utter_greet" in response_names
        assert "utter_goodbye" in response_names
        assert "utter_ask_email" in response_names  # From subdirectory

        # Verify subdirectory response has correct path
        ask_email = next(r for r in result.responses if r.name == "utter_ask_email")
        assert "subdir/user_responses.yml" in ask_email.file_path


class TestListProjectCustomActions:
    """Test list_project_custom_actions function."""

    @pytest.mark.asyncio
    async def test_list_project_custom_actions(self, rasa_project: Path) -> None:
        """Test listing custom actions from domain including subdirectories."""
        # Given: Domain files in main directory and subdirectories

        # When: Listing custom actions
        result = await list_project_custom_actions(str(rasa_project), "domain")

        # Then: Returns custom actions from all subdirectories
        assert result.success is True
        assert result.error is None
        assert result.count == 4

        action_names = [a.name for a in result.actions]
        assert "action_search" in action_names
        assert "action_submit" in action_names
        assert "action_with_config" in action_names  # Dict-style action
        assert "action_validate_email" in action_names  # From subdirectory
        assert "utter_greet" not in action_names  # Filtered out

        # Verify subdirectory action has correct path
        validate_email = next(
            a for a in result.actions if a.name == "action_validate_email"
        )
        assert "subdir/user_actions.yml" in validate_email.file_path


class TestPathTraversalPrevention:
    """Test that path traversal is blocked at the tool level."""

    @pytest.mark.parametrize(
        "malicious_folder",
        ["/etc", "../../etc", "../secret", "data/../../etc"],
        ids=["absolute", "double-dotdot", "single-dotdot", "hidden-in-middle"],
    )
    def test_get_flow_yaml_files_rejects_traversal(
        self, tmp_path: Path, malicious_folder: str
    ) -> None:
        """Test _get_flow_yaml_files rejects path traversal in data_folder."""
        with pytest.raises(InvalidPathException):
            list(_get_flow_yaml_files(tmp_path, malicious_folder))

    @pytest.mark.parametrize(
        "malicious_folder",
        ["/etc", "../../etc", "../secret", "data/../../etc"],
        ids=["absolute", "double-dotdot", "single-dotdot", "hidden-in-middle"],
    )
    def test_get_domain_yaml_files_rejects_traversal(
        self, tmp_path: Path, malicious_folder: str
    ) -> None:
        """Test _get_domain_yaml_files rejects path traversal in domain_folder."""
        with pytest.raises(InvalidPathException):
            list(_get_domain_yaml_files(tmp_path, malicious_folder))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_func",
        [
            list_project_flows,
            list_project_slots,
            list_project_responses,
            list_project_custom_actions,
        ],
        ids=["flows", "slots", "responses", "custom_actions"],
    )
    async def test_high_level_tools_reject_traversal(
        self, tmp_path: Path, tool_func: object
    ) -> None:
        """Test that all high-level project context tools return error on traversal."""
        result = await tool_func(str(tmp_path), "/etc")

        assert result.success is False
        assert result.error is not None
        assert "Path traversal" in result.error


class TestGetProjectFlow:
    """Test get_project_flow function."""

    @pytest.mark.asyncio
    async def test_get_flow_by_id(self, rasa_project: Path) -> None:
        """Get flow by flow ID (YAML key)."""
        result = await get_project_flow(str(rasa_project), "greet_user", "data")
        assert result.success is True
        assert result.flow is not None
        assert result.flow.id == "greet_user"
        assert result.flow.name == "Greet User"
        assert result.flow.definition is not None
        assert "steps" in result.flow.definition
        assert result.flow.definition.get("name") == "Greet User"

    @pytest.mark.asyncio
    async def test_get_flow_by_name(self, rasa_project: Path) -> None:
        """Get flow by human-readable name."""
        result = await get_project_flow(str(rasa_project), "Greet User", "data")
        assert result.success is True
        assert result.flow is not None
        assert result.flow.id == "greet_user"
        assert result.flow.definition is not None

    @pytest.mark.asyncio
    async def test_get_flow_not_found(self, rasa_project: Path) -> None:
        """Unknown flow returns success=False."""
        result = await get_project_flow(str(rasa_project), "nonexistent_flow", "data")
        assert result.success is False
        assert result.flow is None
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_get_flow_empty_id_returns_error(self, rasa_project: Path) -> None:
        """Empty flow_id returns error."""
        result = await get_project_flow(str(rasa_project), "", "data")
        assert result.success is False
        assert result.error is not None


class TestGetProjectSlot:
    """Test get_project_slot function."""

    @pytest.mark.asyncio
    async def test_get_slot_by_name(self, rasa_project: Path) -> None:
        """Get slot by name."""
        result = await get_project_slot(str(rasa_project), "name", "domain")
        assert result.success is True
        assert result.slot is not None
        assert result.slot.name == "name"
        assert result.slot.type == "text"
        assert result.slot.definition is not None
        assert result.slot.definition.get("type") == "text"

    @pytest.mark.asyncio
    async def test_get_slot_not_found(self, rasa_project: Path) -> None:
        """Unknown slot returns success=False."""
        result = await get_project_slot(str(rasa_project), "nonexistent_slot", "domain")
        assert result.success is False
        assert result.slot is None
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_get_slot_empty_name_returns_error(self, rasa_project: Path) -> None:
        """Empty slot_name returns error."""
        result = await get_project_slot(str(rasa_project), "", "domain")
        assert result.success is False
        assert result.error is not None


class TestGetProjectResponse:
    """Test get_project_response function."""

    @pytest.mark.asyncio
    async def test_get_response_by_name(self, rasa_project: Path) -> None:
        """Get response by name."""
        result = await get_project_response(str(rasa_project), "utter_greet", "domain")
        assert result.success is True
        assert result.response is not None
        assert result.response.name == "utter_greet"
        assert result.response.definition is not None
        assert len(result.response.definition) >= 1
        assert result.response.definition[0].get("text") == "Hello!"

    @pytest.mark.asyncio
    async def test_get_response_not_found(self, rasa_project: Path) -> None:
        """Unknown response returns success=False."""
        result = await get_project_response(
            str(rasa_project), "utter_nonexistent", "domain"
        )
        assert result.success is False
        assert result.response is None
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_get_response_empty_name_returns_error(
        self, rasa_project: Path
    ) -> None:
        """Empty response_name returns error."""
        result = await get_project_response(str(rasa_project), "", "domain")
        assert result.success is False
        assert result.error is not None


class TestGetAssistantLogs:
    """Test get_assistant_logs function."""

    @pytest.mark.asyncio
    async def test_get_assistant_logs_returns_string(self) -> None:
        """Test that get_assistant_logs returns a string."""
        result = await get_assistant_logs()

        assert isinstance(result, str)
        # May be empty if no logs have been collected
