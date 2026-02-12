"""Tests for custom actions MCP tool."""

import pytest

from rasa.builder.copilot.mcp_server.tools.custom_actions import (
    list_custom_action_implementations,
)


class TestListCustomActionImplementations:
    @pytest.fixture
    def project_with_actions(self, tmp_path):
        """Create a project with sample custom actions."""
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()

        # Create __init__.py
        (actions_dir / "__init__.py").write_text("")

        # Create a simple action file
        actions_file = actions_dir / "actions.py"
        actions_file.write_text('''"""Custom actions for the bot."""

from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionCheckBalance(Action):
    """Action to check user account balance."""

    def name(self) -> Text:
        return "action_check_balance"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Your balance is $100")
        return []


class ActionTransferMoney(Action):
    def name(self) -> Text:
        return "action_transfer_money"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        return []
''')

        return tmp_path

    @pytest.fixture
    def project_with_nested_actions(self, tmp_path):
        """Create a project with actions in nested directories."""
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")

        # Create nested directory
        transfers_dir = actions_dir / "transfers"
        transfers_dir.mkdir()
        (transfers_dir / "__init__.py").write_text("")

        # Create action in nested directory
        transfer_file = transfers_dir / "transfer_actions.py"
        transfer_file.write_text("""from rasa_sdk import Action


class ActionExecuteTransfer(Action):
    def name(self):
        return "action_execute_transfer"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        return tmp_path

    @pytest.fixture
    def project_without_actions(self, tmp_path):
        """Create a project without actions folder."""
        # Just create an empty project
        (tmp_path / "domain.yml").write_text("version: '3.1'")
        return tmp_path

    @pytest.fixture
    def project_with_empty_actions(self, tmp_path):
        """Create a project with empty actions folder."""
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        return tmp_path

    @pytest.fixture
    def project_with_syntax_error(self, tmp_path):
        """Create a project with a Python file containing syntax errors."""
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")

        # File with syntax error
        (actions_dir / "broken.py").write_text("""
class ActionBroken(Action):
    def name(self)  # Missing colon
        return "action_broken"
""")

        # Valid file alongside broken one
        (actions_dir / "valid.py").write_text("""
from rasa_sdk import Action


class ActionValid(Action):
    def name(self):
        return "action_valid"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        return tmp_path

    @pytest.mark.asyncio
    async def test_finds_actions_in_folder(self, project_with_actions):
        """Test that actions are found in the actions folder."""
        result = await list_custom_action_implementations(str(project_with_actions))

        assert result.error is None
        assert result.count == 2
        assert result.actions_folder == "actions"
        assert len(result.actions) == 2

        names = {a.name for a in result.actions}
        assert "action_check_balance" in names
        assert "action_transfer_money" in names

    @pytest.mark.asyncio
    async def test_extracts_class_names(self, project_with_actions):
        """Test that class names are correctly extracted."""
        result = await list_custom_action_implementations(str(project_with_actions))

        class_names = {a.class_name for a in result.actions}
        assert "ActionCheckBalance" in class_names
        assert "ActionTransferMoney" in class_names

    @pytest.mark.asyncio
    async def test_extracts_file_paths(self, project_with_actions):
        """Test that file paths are correctly extracted."""
        result = await list_custom_action_implementations(str(project_with_actions))

        for action in result.actions:
            assert action.file_path == "actions/actions.py"

    @pytest.mark.asyncio
    async def test_extracts_basic_info(self, project_with_actions):
        """Test that basic action information is correctly extracted."""
        result = await list_custom_action_implementations(str(project_with_actions))

        for action in result.actions:
            assert action.class_name is not None
            assert len(action.class_name) > 0
            assert action.file_path is not None
            assert action.file_path.endswith(".py")

    @pytest.mark.asyncio
    async def test_finds_actions_in_nested_directories(
        self, project_with_nested_actions
    ):
        """Test that actions in nested directories are found."""
        result = await list_custom_action_implementations(
            str(project_with_nested_actions)
        )

        assert result.error is None
        assert result.count == 1
        assert result.actions[0].class_name == "ActionExecuteTransfer"
        assert result.actions[0].name == "action_execute_transfer"
        assert "transfers/transfer_actions.py" in result.actions[0].file_path

    @pytest.mark.asyncio
    async def test_returns_error_when_no_actions_folder(self, project_without_actions):
        """Test that error is returned when actions folder doesn't exist."""
        result = await list_custom_action_implementations(str(project_without_actions))

        assert result.error is not None
        assert "not found" in result.error.lower()
        assert "actions_folder" in result.error or "endpoints.yml" in result.error
        assert result.count == 0
        assert result.actions == []
        assert result.actions_folder == "actions"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_actions_defined(
        self, project_with_empty_actions
    ):
        """Test that empty result is returned when no actions are defined."""
        result = await list_custom_action_implementations(
            str(project_with_empty_actions)
        )

        assert result.error is None
        assert result.count == 0
        assert result.actions == []

    @pytest.mark.asyncio
    async def test_handles_syntax_errors_gracefully(self, project_with_syntax_error):
        """Test that syntax errors in Python files don't break the scan."""
        result = await list_custom_action_implementations(
            str(project_with_syntax_error)
        )

        # Should still find the valid action
        assert result.error is None
        assert result.count == 1
        assert result.actions[0].class_name == "ActionValid"
        assert result.actions[0].name == "action_valid"

    @pytest.mark.asyncio
    async def test_custom_actions_folder_name(self, tmp_path):
        """Test that custom actions folder name can be specified."""
        # Create custom folder name
        custom_dir = tmp_path / "my_actions"
        custom_dir.mkdir()
        (custom_dir / "__init__.py").write_text("")
        (custom_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionCustom(Action):
    def name(self):
        return "action_custom"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        result = await list_custom_action_implementations(
            str(tmp_path), actions_folder="my_actions"
        )

        assert result.error is None
        assert result.count == 1
        assert result.actions_folder == "my_actions"
        assert result.actions[0].class_name == "ActionCustom"

    @pytest.mark.asyncio
    async def test_ignores_non_action_classes(self, tmp_path):
        """Test that non-Action classes are ignored."""
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        (actions_dir / "helpers.py").write_text('''
from rasa_sdk import Action


class HelperClass:
    """This is not an action."""
    pass


class AnotherHelper:
    def name(self):
        return "not_an_action"


class ActionReal(Action):
    def name(self):
        return "action_real"

    def run(self, dispatcher, tracker, domain):
        return []
''')

        result = await list_custom_action_implementations(str(tmp_path))

        assert result.error is None
        assert result.count == 1
        assert result.actions[0].class_name == "ActionReal"

    @pytest.mark.asyncio
    async def test_handles_action_subclass_variations(self, tmp_path):
        """Test that various Action inheritance patterns are recognized."""
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        (actions_dir / "various.py").write_text("""
from rasa_sdk import Action
from rasa_sdk.knowledge_base.actions import ActionQueryKnowledgeBase


class ActionSimple(Action):
    def name(self):
        return "action_simple"

    def run(self, dispatcher, tracker, domain):
        return []


class ActionKB(ActionQueryKnowledgeBase):
    def name(self):
        return "action_kb"
""")

        result = await list_custom_action_implementations(str(tmp_path))

        assert result.error is None
        assert result.count == 2
        class_names = {a.class_name for a in result.actions}
        assert "ActionSimple" in class_names
        assert "ActionKB" in class_names


class TestListCustomActionsServerTool:
    """Test the MCP server tool wrapper."""

    @pytest.fixture
    def mock_project_folder(self, monkeypatch, tmp_path):
        """Set up mock project folder."""
        monkeypatch.setenv("RASA_PROJECT_FOLDER", str(tmp_path))

        # Create actions folder with sample action
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        (actions_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionTest(Action):
    def name(self):
        return "action_test"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        return tmp_path

    @pytest.mark.asyncio
    async def test_list_custom_actions_tool(self, mock_project_folder):
        """Test the list_custom_actions tool from server.py."""
        from rasa.builder.copilot.mcp_server.server import list_custom_actions

        result = await list_custom_actions()

        assert result.error is None
        assert result.count == 1
        assert result.actions[0].class_name == "ActionTest"
        assert result.actions[0].name == "action_test"

    @pytest.mark.asyncio
    async def test_list_custom_actions_tool_with_custom_folder(
        self, mock_project_folder
    ):
        """Test the tool with a custom actions folder name."""
        from rasa.builder.copilot.mcp_server.server import list_custom_actions

        # Create custom folder
        custom_dir = mock_project_folder / "custom_actions"
        custom_dir.mkdir()
        (custom_dir / "__init__.py").write_text("")
        (custom_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionCustomFolder(Action):
    def name(self):
        return "action_custom_folder"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        result = await list_custom_actions(actions_folder="custom_actions")

        assert result.error is None
        assert result.count == 1
        assert result.actions_folder == "custom_actions"
        assert result.actions[0].class_name == "ActionCustomFolder"


class TestActionsModuleAutoDetection:
    """Test auto-detection of actions_module from endpoints.yml."""

    @pytest.mark.asyncio
    async def test_detects_actions_module_from_endpoints_yml(self, tmp_path):
        """Test that actions_module is auto-detected from endpoints.yml."""
        # Create endpoints.yml with custom actions_module
        endpoints_content = """
action_endpoint:
  actions_module: "my_custom_actions"
"""
        (tmp_path / "endpoints.yml").write_text(endpoints_content)

        # Create the custom actions folder
        custom_dir = tmp_path / "my_custom_actions"
        custom_dir.mkdir()
        (custom_dir / "__init__.py").write_text("")
        (custom_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionFromEndpoints(Action):
    def name(self):
        return "action_from_endpoints"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        result = await list_custom_action_implementations(str(tmp_path))

        assert result.error is None
        assert result.count == 1
        assert result.actions_folder == "my_custom_actions"
        assert result.actions[0].class_name == "ActionFromEndpoints"

    @pytest.mark.asyncio
    async def test_detects_from_endpoints_yaml_extension(self, tmp_path):
        """Test detection from endpoints.yaml (alternative extension)."""
        endpoints_content = """
action_endpoint:
  actions_module: "yaml_actions"
"""
        (tmp_path / "endpoints.yaml").write_text(endpoints_content)

        # Create the actions folder
        actions_dir = tmp_path / "yaml_actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        (actions_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionYaml(Action):
    def name(self):
        return "action_yaml"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        result = await list_custom_action_implementations(str(tmp_path))

        assert result.error is None
        assert result.actions_folder == "yaml_actions"

    @pytest.mark.asyncio
    async def test_explicit_folder_overrides_endpoints_yml(self, tmp_path):
        """Test that explicit actions_folder parameter overrides endpoints.yml."""
        # Create endpoints.yml with one module
        endpoints_content = """
action_endpoint:
  actions_module: "from_endpoints"
"""
        (tmp_path / "endpoints.yml").write_text(endpoints_content)

        # Create folder from endpoints.yml
        endpoints_dir = tmp_path / "from_endpoints"
        endpoints_dir.mkdir()
        (endpoints_dir / "__init__.py").write_text("")
        (endpoints_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionFromEndpoints(Action):
    def name(self):
        return "action_from_endpoints"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        # Create explicit folder
        explicit_dir = tmp_path / "explicit_folder"
        explicit_dir.mkdir()
        (explicit_dir / "__init__.py").write_text("")
        (explicit_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionExplicit(Action):
    def name(self):
        return "action_explicit"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        # Explicitly specify folder - should override endpoints.yml
        result = await list_custom_action_implementations(
            str(tmp_path), actions_folder="explicit_folder"
        )

        assert result.error is None
        assert result.actions_folder == "explicit_folder"
        assert result.actions[0].class_name == "ActionExplicit"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_when_no_endpoints(self, tmp_path):
        """Test fallback to 'actions' when no endpoints.yml exists."""
        # Create only the default actions folder
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        (actions_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionDefault(Action):
    def name(self):
        return "action_default"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        result = await list_custom_action_implementations(str(tmp_path))

        assert result.error is None
        assert result.actions_folder == "actions"
        assert result.actions[0].class_name == "ActionDefault"

    @pytest.mark.asyncio
    async def test_handles_endpoints_without_actions_module(self, tmp_path):
        """Test handling endpoints.yml without actions_module configured."""
        # Create endpoints.yml without actions_module
        endpoints_content = """
action_endpoint:
  url: "http://localhost:5055/webhook"
"""
        (tmp_path / "endpoints.yml").write_text(endpoints_content)

        # Create default actions folder
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        (actions_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionNoModule(Action):
    def name(self):
        return "action_no_module"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        result = await list_custom_action_implementations(str(tmp_path))

        assert result.error is None
        # Should fall back to default "actions"
        assert result.actions_folder == "actions"
        assert result.actions[0].class_name == "ActionNoModule"

    @pytest.mark.asyncio
    async def test_handles_malformed_endpoints_yml(self, tmp_path):
        """Test graceful handling of malformed endpoints.yml."""
        # Create malformed endpoints.yml
        (tmp_path / "endpoints.yml").write_text("not: valid: yaml: content: [")

        # Create default actions folder
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        (actions_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionMalformed(Action):
    def name(self):
        return "action_malformed"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        result = await list_custom_action_implementations(str(tmp_path))

        # Should not fail, just fall back to default
        assert result.error is None
        assert result.actions_folder == "actions"

    @pytest.mark.asyncio
    async def test_converts_dotted_module_path_to_filesystem_path(self, tmp_path):
        """Test that dotted module paths are converted to filesystem paths."""
        # Create endpoints.yml with dotted module path
        endpoints_content = """
action_endpoint:
  actions_module: "data.dummy_actions_module"
"""
        (tmp_path / "endpoints.yml").write_text(endpoints_content)

        # Create the nested actions folder matching the dotted path
        nested_dir = tmp_path / "data" / "dummy_actions_module"
        nested_dir.mkdir(parents=True)
        (nested_dir / "__init__.py").write_text("")
        (nested_dir / "actions.py").write_text("""
from rasa_sdk import Action


class ActionDotted(Action):
    def name(self):
        return "action_dotted"

    def run(self, dispatcher, tracker, domain):
        return []
""")

        result = await list_custom_action_implementations(str(tmp_path))

        assert result.error is None
        assert result.count == 1
        assert result.actions_folder == "data/dummy_actions_module"
        assert result.actions[0].class_name == "ActionDotted"
        assert result.actions[0].name == "action_dotted"

    @pytest.mark.asyncio
    async def test_error_reports_auto_detected_folder(self, tmp_path):
        """Test that errors report the auto-detected folder, not the default."""
        # Create endpoints.yml with custom module
        endpoints_content = """
action_endpoint:
  actions_module: "custom_module"
"""
        (tmp_path / "endpoints.yml").write_text(endpoints_content)

        # Don't create the folder - this will trigger the "not found" error
        result = await list_custom_action_implementations(str(tmp_path))

        # Error should report the auto-detected folder "custom_module", not "actions"
        assert result.error is not None
        assert result.actions_folder == "custom_module"
        assert "custom_module" in result.error
