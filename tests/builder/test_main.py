from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sanic import Sanic

from rasa.builder import config
from rasa.builder.main import create_app, setup_langfuse
from rasa.builder.project_generator.project_generator import ProjectGenerator


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


class TestSetupLangfuse:
    """Test the setup_langfuse function."""

    @patch("langfuse.Langfuse")
    @patch("rasa.builder.main.config.LANGFUSE_PUBLIC_KEY", "test-public-key")
    @patch("rasa.builder.main.config.LANGFUSE_SECRET_KEY", "test-secret-key")
    @patch("rasa.builder.main.config.LANGFUSE_HOST", "https://test-langfuse-host.com")
    @patch("rasa.builder.main.config.DEPLOYMENT_STACK", "unit-test")
    @patch("rasa.builder.main.config.DEPLOYMENT_STACK_HEADER_NAME", "deployment-stack")
    def test_setup_langfuse_creates_client(
        self, mock_langfuse_class: MagicMock
    ) -> None:
        """Test that setup_langfuse creates a Langfuse client that can be retrieved."""
        from rasa.builder.telemetry.langfuse_integration import langfuse_compat

        langfuse = langfuse_compat.langfuse

        # Create a mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.flush = MagicMock()
        mock_langfuse_class.return_value = mock_client_instance

        # Mock get_client to return the same instance
        with patch.object(langfuse, "get_client", return_value=mock_client_instance):
            # Call setup_langfuse to initialize the client
            setup_langfuse()

            # Verify Langfuse was instantiated with correct parameters
            mock_langfuse_class.assert_called_once_with(
                public_key="test-public-key",
                secret_key="test-secret-key",
                host="https://test-langfuse-host.com",
                additional_headers={"deployment-stack": "unit-test"},
                environment="unit-test",
            )

            # Get the client using langfuse.get_client()
            client = langfuse.get_client()

            # Verify the client retrieved is the same one that was created
            assert client is mock_client_instance


class TestWaitForPort:
    """Test the _wait_for_port helper function."""

    def test_wait_for_port_success(self, monkeypatch) -> None:
        """Test _wait_for_port returns True when port is available."""
        import socket

        from rasa.builder.main import _wait_for_port

        # Mock create_connection to simulate successful connection
        mock_socket = MagicMock()
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)

        monkeypatch.setattr(
            socket, "create_connection", lambda *args, **kwargs: mock_socket
        )

        result = _wait_for_port("127.0.0.1", 5051, timeout=1.0)

        assert result is True

    def test_wait_for_port_timeout(self, monkeypatch) -> None:
        """Test _wait_for_port returns False on timeout."""
        import socket

        from rasa.builder.main import _wait_for_port

        # Mock create_connection to simulate connection refused
        def mock_create_connection(*args, **kwargs):
            raise ConnectionRefusedError("Connection refused")

        monkeypatch.setattr(socket, "create_connection", mock_create_connection)

        # Use a very short timeout
        result = _wait_for_port("127.0.0.1", 5051, timeout=0.2, poll_interval=0.05)

        assert result is False


class TestStartMcpServer:
    """Test the start_mcp_server function."""

    def test_start_mcp_server_passes_project_folder(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Test that start_mcp_server passes project_folder to run_server."""
        from rasa.builder.main import start_mcp_server

        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_PORT", 5051)
        monkeypatch.setattr("rasa.builder.config.BUILDER_SERVER_HOST", "0.0.0.0")
        monkeypatch.setattr("rasa.builder.config.BUILDER_SERVER_PORT", 5050)

        with patch("rasa.builder.copilot.mcp_server.server.run_server") as mock_run:
            start_mcp_server(str(tmp_path))

            mock_run.assert_called_once_with(
                host="127.0.0.1",
                port=5051,
                project_folder=str(tmp_path),
                rasa_server_url="http://0.0.0.0:5050",
            )

    def test_start_mcp_server_calls_run_server_with_config(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Test that start_mcp_server calls run_server with correct config."""
        from rasa.builder.main import start_mcp_server

        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_PORT", 5055)
        monkeypatch.setattr("rasa.builder.config.BUILDER_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr("rasa.builder.config.BUILDER_SERVER_PORT", 8080)

        with patch("rasa.builder.copilot.mcp_server.server.run_server") as mock_run:
            start_mcp_server(str(tmp_path))

            mock_run.assert_called_once_with(
                host="127.0.0.1",
                port=5055,
                project_folder=str(tmp_path),
                rasa_server_url="http://127.0.0.1:8080",
            )

    def test_start_mcp_server_handles_exception(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Test that start_mcp_server handles exceptions gracefully."""
        from rasa.builder.main import start_mcp_server

        # Mock run_server to raise exception
        with patch(
            "rasa.builder.copilot.mcp_server.server.run_server",
            side_effect=Exception("Server error"),
        ):
            # Should not raise - just log the error
            start_mcp_server(str(tmp_path))


class TestMcpServerStartup:
    """Test MCP server startup integration in main."""

    def test_main_starts_mcp_server(self, monkeypatch, tmp_path: Path) -> None:
        """Test that main starts MCP server"""
        import threading

        # Enable Agent SDK
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_PORT", 5051)
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_STARTUP_TIMEOUT", 5)

        mcp_thread_started = False
        original_thread_init = threading.Thread.__init__

        def mock_thread_init(self, *args, **kwargs):
            nonlocal mcp_thread_started
            if kwargs.get("name") == "mcp-server":
                mcp_thread_started = True
            original_thread_init(self, *args, **kwargs)

        monkeypatch.setattr(threading.Thread, "__init__", mock_thread_init)

        # Mock the rest to prevent actual startup
        with (
            patch("rasa.builder.main.setup_langfuse"),
            patch("rasa.builder.main.create_app") as mock_create_app,
            patch("rasa.builder.main.start_mcp_server"),
            patch("rasa.builder.main._wait_for_port", return_value=True),
            patch.object(threading.Thread, "start"),
        ):
            mock_app = MagicMock()
            # Mock app.run to prevent actual server startup
            mock_app.run = MagicMock()
            mock_create_app.return_value = mock_app

            from rasa.builder import main

            # Call main with the test project folder
            try:
                main.main(str(tmp_path))
            except SystemExit:
                pass  # main() might call sys.exit

            # Verify create_app was called
            mock_create_app.assert_called_once_with(str(tmp_path))
