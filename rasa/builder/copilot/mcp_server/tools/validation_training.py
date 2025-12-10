"""MCP tools for validation and training operations.

These tools allow the agent to validate project files and train models,
enabling an iterative development loop.
"""

from pathlib import Path

import aiohttp
import structlog

from rasa.builder.copilot.mcp_server.models import (
    TrainingResponse,
    ValidationErrorDetail,
    ValidationResponse,
)
from rasa.builder.exceptions import TrainingError
from rasa.builder.exceptions import ValidationError as RasaValidationError

# File name constants
DOMAIN_FILE = "domain.yml"
CONFIG_FILE = "config.yml"
CONFIG_FILE_YAML = "config.yaml"
DOMAIN_DIR = "domain"

structlogger = structlog.get_logger()


async def _notify_sanic_to_reload_agent() -> bool:
    """Notify the main Sanic server to reload the agent.

    This calls an internal endpoint on the Sanic server to trigger
    agent reload after successful training. The endpoint is protected
    and only accessible from localhost.

    Returns:
        True if the agent was reloaded successfully, False otherwise.
    """
    from rasa.builder import config

    url = (
        f"http://{config.BUILDER_SERVER_HOST}:{config.BUILDER_SERVER_PORT}"
        f"/api/internal/reload-agent"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    structlogger.info(
                        "mcp_server.tools.validation_training.agent_reload_success",
                        event_info="Notified Sanic server to reload agent successfully",
                    )
                    return True
                else:
                    body = await resp.text()
                    structlogger.warning(
                        "mcp_server.tools.validation_training.agent_reload_failed",
                        event_info="Failed to notify Sanic server to reload agent",
                        status=resp.status,
                        response=body,
                    )
                    return False
    except Exception as e:
        structlogger.warning(
            "mcp_server.tools.validation_training.agent_reload_error",
            event_info="Error notifying Sanic server to reload agent",
            error=str(e),
        )
        return False


async def validate_assistant_project(project_folder: str) -> ValidationResponse:
    """Validate the bot project configuration.

    This runs `rasa data validate` on the current project and returns any errors.

    Args:
        project_folder: Path to the project folder

    Returns:
        ValidationResponse with validation results
    """
    try:
        # Lazy import heavy Rasa modules only when needed
        from rasa.builder.validation_service import validate_project
        from rasa.shared.importers.importer import TrainingDataImporter

        structlogger.info(
            "mcp_server.tools.validation_training.validate_assistant_project",
            event_info="Running validation",
        )

        # Create importer (same logic as ProjectGenerator._create_importer)
        project_path = Path(project_folder)

        # Determine domain path
        if (project_path / DOMAIN_FILE).exists():
            domain_path = str(project_path / DOMAIN_FILE)
        else:
            domain_path = str(project_path / DOMAIN_DIR)

        # Determine config path
        if (project_path / CONFIG_FILE).exists():
            config_path = str(project_path / CONFIG_FILE)
        else:
            config_path = str(project_path / CONFIG_FILE_YAML)

        # Create importer
        importer = TrainingDataImporter.load_from_config(
            config_path=config_path,
            domain_path=domain_path,
            training_data_paths=[str(project_path / "data")],
            args={},
        )

        # Run validation
        result = await validate_project(importer)

        if result is None:
            # Validation passed
            return ValidationResponse(
                success=True,
                errors=None,
                message="Validation passed successfully",
            )
        else:
            # Validation failed with message
            return ValidationResponse(
                success=False,
                errors=[ValidationErrorDetail(message=result)],
                message="Validation failed",
            )

    except RasaValidationError as e:
        # Extract validation logs for detailed error info
        errors = []
        if hasattr(e, "validation_logs") and e.validation_logs:
            for log in e.validation_logs:
                errors.append(
                    ValidationErrorDetail(
                        level=log.get("log_level", "error"),
                        message=log.get("event", ""),
                        details={
                            k: v
                            for k, v in log.items()
                            if k not in ["log_level", "event"]
                        },
                    )
                )

        return ValidationResponse(
            success=False,
            errors=errors if errors else [ValidationErrorDetail(message=str(e))],
            message=f"Validation failed: {e!s}",
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.validation_training.validate_assistant_project.error",
            event_info="Validation failed with exception",
            error=str(e),
        )
        return ValidationResponse(
            success=False,
            errors=[ValidationErrorDetail(message=str(e))],
            message=f"Validation error: {e!s}",
        )


async def train_assistant(project_folder: str) -> TrainingResponse:
    """Train the bot model.

    This runs `rasa train` on the current project.

    Args:
        project_folder: Path to the project folder

    Returns:
        TrainingResponse with training results
    """
    try:
        # Lazy import heavy Rasa modules only when needed
        from rasa.builder.models import TrainingInput
        from rasa.builder.training_service import train_and_load_agent
        from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH
        from rasa.shared.importers.importer import TrainingDataImporter

        structlogger.info(
            "mcp_server.tools.validation_training.train_assistant",
            event_info="Starting model training",
        )

        # Get paths
        project_path = Path(project_folder)
        endpoints_file = project_path / DEFAULT_ENDPOINTS_PATH

        # Determine config path
        if (project_path / CONFIG_FILE).exists():
            config_file = project_path / CONFIG_FILE
        else:
            config_file = project_path / CONFIG_FILE_YAML

        # Determine domain path
        if (project_path / DOMAIN_FILE).exists():
            domain_path = str(project_path / DOMAIN_FILE)
        else:
            domain_path = str(project_path / DOMAIN_DIR)

        # Create importer
        importer = TrainingDataImporter.load_from_config(
            config_path=str(config_file),
            domain_path=domain_path,
            training_data_paths=[str(project_path / "data")],
            args={},
        )

        # Create training input
        training_input = TrainingInput(
            importer=importer,
            endpoints_file=endpoints_file,
            config_file=config_file,
        )

        # Train and load agent
        agent = await train_and_load_agent(
            training_input, role="copilot", action="generation"
        )

        # Get model path from agent
        model_path = None
        if hasattr(agent, "model_name") and agent.model_name:
            model_path = str(project_path / "models" / agent.model_name)

        # Notify Sanic server to reload the agent
        agent_reloaded = await _notify_sanic_to_reload_agent()

        return TrainingResponse(
            success=True,
            model_path=model_path,
            message="Model training completed successfully",
            agent_reloaded=agent_reloaded,
        )

    except TrainingError as e:
        structlogger.error(
            "mcp_server.tools.validation_training.train_assistant.training_error",
            event_info="Training failed",
            error=str(e),
        )
        return TrainingResponse(
            success=False,
            model_path=None,
            message=f"Training failed: {e!s}",
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.validation_training.train_assistant.error",
            event_info="Training failed with exception",
            error=str(e),
        )
        return TrainingResponse(
            success=False,
            model_path=None,
            message=f"Training error: {e!s}",
        )
