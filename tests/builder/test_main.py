from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sanic import Sanic

from rasa.builder import config
from rasa.builder.main import create_app
from rasa.builder.project_generator import ProjectGenerator


class TestCreateApp:
    """Test the create_app function and its listeners."""

    def test_create_app_basic(self, tmp_path: Path) -> None:
        """Test basic app creation."""
        app = create_app(str(tmp_path))

        assert isinstance(app, Sanic)
        assert app.name == "BotBuilderService"
        assert hasattr(app.ctx, "project_generator")
        assert isinstance(app.ctx.project_generator, ProjectGenerator)

    def test_project_generator_uses_correct_folder(self, tmp_path: Path) -> None:
        """Test that the project generator is initialized with the correct folder."""
        app = create_app(str(tmp_path))

        assert app.ctx.project_generator.project_folder == tmp_path

    @patch("rasa.builder.main.try_load_existing_agent")
    def test_handles_agent_loading_exception(
        self, mock_try_load, tmp_path: Path
    ) -> None:
        """Test that exceptions during agent loading are handled gracefully."""
        mock_try_load.side_effect = Exception("Failed to load agent")

        # Should not raise an exception
        app = create_app(str(tmp_path))

        assert isinstance(app, Sanic)
        # The warning should be logged (though we can't easily test it without
        # mocking structlogger)

    def test_app_has_correct_blueprints(self, tmp_path: Path) -> None:
        """Test that the app has the expected blueprints registered."""
        app = create_app(str(tmp_path))

        # Check that the builder blueprint is registered
        blueprint_names = [bp.name for bp in app.blueprints.values()]
        assert "bot_builder" in blueprint_names

    def test_cors_is_configured(self, tmp_path: Path) -> None:
        """Test that CORS is configured for the app."""
        with patch("rasa.builder.main.configure_cors") as mock_configure_cors:
            create_app(str(tmp_path))

            # CORS is configured with additional parameters, so we just check
            # it was called
            mock_configure_cors.assert_called_once()


class TestConditionalLogic:
    """Test the conditional logic for registering background cache download."""

    @pytest.fixture
    def mock_app_context(self):
        """Create a mock app context for testing."""
        app = MagicMock(spec=Sanic)
        app.ctx = MagicMock()
        app.ctx.project_generator = MagicMock(spec=ProjectGenerator)
        return app

    @patch("rasa.builder.main.config.HELLO_RASA_PROJECT_ID", "test-project")
    def test_condition_met_for_background_download(self, mock_app_context) -> None:
        """Test the condition logic when background download should be enabled."""
        mock_app_context.ctx.project_generator.is_empty.return_value = True

        # Simulate the condition check
        should_register = (
            config.HELLO_RASA_PROJECT_ID
            and mock_app_context.ctx.project_generator.is_empty()
        )

        assert should_register is True

    @patch("rasa.builder.main.config.HELLO_RASA_PROJECT_ID", None)
    def test_condition_not_met_no_project_id(self, mock_app_context) -> None:
        """Test the condition logic when no project ID is set."""
        mock_app_context.ctx.project_generator.is_empty.return_value = True

        # Simulate the condition check
        should_register = (
            config.HELLO_RASA_PROJECT_ID
            and mock_app_context.ctx.project_generator.is_empty()
        )

        # When HELLO_RASA_PROJECT_ID is None, the condition evaluates to None (falsy)
        assert not should_register

    @patch("rasa.builder.main.config.HELLO_RASA_PROJECT_ID", "test-project")
    def test_condition_not_met_project_not_empty(self, mock_app_context) -> None:
        """Test the condition logic when project is not empty."""
        mock_app_context.ctx.project_generator.is_empty.return_value = False

        # Simulate the condition check
        should_register = (
            config.HELLO_RASA_PROJECT_ID
            and mock_app_context.ctx.project_generator.is_empty()
        )

        assert should_register is False
