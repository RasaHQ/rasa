"""Tests for MCP validation and training tools."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder.copilot.mcp_server.models import (
    TrainingResponse,
    ValidationResponse,
)
from rasa.builder.copilot.mcp_server.tools.validation_training import (
    _notify_sanic_to_reload_agent,
    train_assistant,
    validate_assistant_project,
)


class TestValidateAssistantProject:
    """Test validate_assistant_project function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a basic project folder structure."""
        (tmp_path / "domain.yml").write_text(
            "version: '3.1'\nintents: []\nresponses: {}"
        )
        (tmp_path / "config.yml").write_text("pipeline: []\npolicies: []")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "nlu.yml").write_text("version: '3.1'\nnlu: []")
        return tmp_path

    @pytest.mark.asyncio
    async def test_validate_project_success(self, project_folder: Path):
        """Test successful project validation."""
        with (
            patch("rasa.builder.validation_service.validate_project") as mock_validate,
            patch(
                "rasa.shared.importers.importer.TrainingDataImporter"
            ) as mock_importer,
        ):
            mock_importer.load_from_config.return_value = MagicMock()
            mock_validate.return_value = None  # None means success

            result = await validate_assistant_project(str(project_folder))

            assert isinstance(result, ValidationResponse)
            assert result.success is True
            assert result.errors is None

    @pytest.mark.asyncio
    async def test_validate_project_with_errors(self, project_folder: Path):
        """Test validation with errors."""
        from rasa.builder.exceptions import ValidationError as RasaValidationError

        with (
            patch("rasa.builder.validation_service.validate_project") as mock_validate,
            patch(
                "rasa.shared.importers.importer.TrainingDataImporter"
            ) as mock_importer,
        ):
            mock_importer.load_from_config.return_value = MagicMock()
            mock_validate.side_effect = RasaValidationError(
                "Domain missing required slots"
            )

            result = await validate_assistant_project(str(project_folder))

            assert result.success is False
            assert result.errors is not None
            assert len(result.errors) == 1
            assert "missing required slots" in result.errors[0].message.lower()

    @pytest.mark.asyncio
    async def test_validate_project_exception(self, project_folder: Path):
        """Test validation when an exception is raised."""
        with patch(
            "rasa.shared.importers.importer.TrainingDataImporter"
        ) as mock_importer:
            mock_importer.load_from_config.side_effect = Exception("Import failed")

            result = await validate_assistant_project(str(project_folder))

            assert result.success is False
            assert result.errors is not None
            assert "Import failed" in result.errors[0].message

    @pytest.mark.asyncio
    async def test_validate_project_validation_error(self, project_folder: Path):
        """Test validation with RasaValidationError."""
        from rasa.builder.exceptions import ValidationError as RasaValidationError

        validation_logs = [{"log_level": "error", "event": "Missing intent definition"}]

        with (
            patch("rasa.builder.validation_service.validate_project") as mock_validate,
            patch(
                "rasa.shared.importers.importer.TrainingDataImporter"
            ) as mock_importer,
        ):
            mock_importer.load_from_config.return_value = MagicMock()
            error = RasaValidationError("Validation failed")
            error.validation_logs = validation_logs
            mock_validate.side_effect = error

            result = await validate_assistant_project(str(project_folder))

            assert result.success is False
            assert result.errors is not None
            assert any("Missing intent" in e.message for e in result.errors)


class TestTrainAssistant:
    """Test train_assistant function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a basic project folder structure."""
        (tmp_path / "domain.yml").write_text(
            "version: '3.1'\nintents: []\nresponses: {}"
        )
        (tmp_path / "config.yml").write_text("pipeline: []\npolicies: []")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "nlu.yml").write_text("version: '3.1'\nnlu: []")
        return tmp_path

    @pytest.mark.asyncio
    async def test_train_assistant_success(self, project_folder: Path):
        """Test successful training."""
        mock_agent = MagicMock()
        mock_agent.model_name = "test-model.tar.gz"

        with (
            patch(
                "rasa.builder.training_service.train_and_load_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "rasa.shared.importers.importer.TrainingDataImporter"
            ) as mock_importer,
            patch("rasa.builder.models.TrainingInput") as mock_training_input,
            patch(
                "rasa.builder.copilot.mcp_server.tools.validation_training._notify_sanic_to_reload_agent",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_importer.load_from_config.return_value = MagicMock()
            mock_training_input.return_value = MagicMock()
            result = await train_assistant(str(project_folder))

            assert isinstance(result, TrainingResponse)
            assert result.success is True
            assert result.model_path is not None
            assert "test-model.tar.gz" in result.model_path
            assert result.agent_reloaded is True

    @pytest.mark.asyncio
    async def test_train_assistant_failure(self, project_folder: Path):
        """Test training failure."""
        from rasa.builder.exceptions import TrainingError

        with patch(
            "rasa.shared.importers.importer.TrainingDataImporter"
        ) as mock_importer:
            mock_importer.load_from_config.side_effect = TrainingError(
                "Training failed: Invalid configuration"
            )

            result = await train_assistant(str(project_folder))

            assert result.success is False
            assert result.model_path is None
            assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_train_assistant_agent_reload_failure(self, project_folder: Path):
        """Test training success but agent reload failure."""
        mock_agent = MagicMock()
        mock_agent.model_name = "test-model.tar.gz"

        with (
            patch(
                "rasa.builder.training_service.train_and_load_agent",
                new_callable=AsyncMock,
                return_value=mock_agent,
            ),
            patch(
                "rasa.shared.importers.importer.TrainingDataImporter"
            ) as mock_importer,
            patch("rasa.builder.models.TrainingInput") as mock_training_input,
            patch(
                "rasa.builder.copilot.mcp_server.tools.validation_training._notify_sanic_to_reload_agent",
                new_callable=AsyncMock,
                return_value=False,  # Reload failed
            ),
        ):
            mock_importer.load_from_config.return_value = MagicMock()
            mock_training_input.return_value = MagicMock()
            result = await train_assistant(str(project_folder))

            # Training still succeeded
            assert result.success is True
            assert result.agent_reloaded is False


class TestNotifySanicToReloadAgent:
    """Test _notify_sanic_to_reload_agent function."""

    @pytest.mark.asyncio
    async def test_notify_reload_success(self):
        """Test successful agent reload notification."""
        mock_response = MagicMock()
        mock_response.status = 200

        with patch(
            "aiohttp.ClientSession.post",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(),
            ),
        ):
            with patch("aiohttp.ClientSession", MagicMock()):
                # Mock the entire session
                mock_session = MagicMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock()

                mock_ctx = MagicMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
                mock_ctx.__aexit__ = AsyncMock()
                mock_session.post = MagicMock(return_value=mock_ctx)

                with patch("aiohttp.ClientSession", return_value=mock_session):
                    result = await _notify_sanic_to_reload_agent()
                    assert result is True

    @pytest.mark.asyncio
    async def test_notify_reload_failure_status(self):
        """Test agent reload notification with non-200 status."""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_ctx)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _notify_sanic_to_reload_agent()
            assert result is False

    @pytest.mark.asyncio
    async def test_notify_reload_connection_error(self):
        """Test agent reload notification with connection error."""
        import aiohttp
        from aioresponses import aioresponses

        with aioresponses() as m:
            # Mock reload endpoint to raise connection error
            m.post(
                "http://localhost:5002/api/internal/reload-agent",
                exception=aiohttp.ClientError("Connection refused"),
            )

            result = await _notify_sanic_to_reload_agent()
            assert result is False
