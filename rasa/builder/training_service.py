"""Functions for training and loading Rasa models."""

import os
from pathlib import Path
from typing import Optional

import structlog
from sanic import Sanic

from rasa.builder.exceptions import AgentLoadError, TrainingError
from rasa.builder.models import TrainingInput
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.studio_chat import StudioChatInput
from rasa.core.config.configuration import Configuration
from rasa.model import get_latest_model
from rasa.model_training import TrainingResult, train
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH
from rasa.shared.importers.importer import TrainingDataImporter

structlogger = structlog.get_logger()


def update_agent(agent: Optional[Agent], app: Sanic) -> None:
    """Update the agent in the request context."""
    app.ctx.agent = agent
    if hasattr(app.ctx, "input_channel") and isinstance(
        app.ctx.input_channel, StudioChatInput
    ):
        app.ctx.input_channel.agent = agent


async def train_and_load_agent(input: TrainingInput) -> Agent:
    """Train a model and load an agent.

    Args:
        input: Training input with importer and endpoints file

    Returns:
        Loaded and ready agent

    Raises:
        TrainingError: If training fails
        AgentLoadError: If agent loading fails
    """
    try:
        # Train the model
        training_result = await _train_model(
            input.importer, input.endpoints_file, input.config_file
        )

        # Load the agent
        agent_instance = await _load_agent(training_result.model, input.endpoints_file)

        # Verify agent is ready
        if not agent_instance.is_ready():
            raise AgentLoadError("Agent failed to load properly - model is not ready")

        structlogger.info("training.agent_ready", model_path=training_result.model)

        return agent_instance

    except (TrainingError, AgentLoadError):
        raise
    except Exception as e:
        raise TrainingError(f"Unexpected error during training: {e}")
    except SystemExit as e:
        raise TrainingError(f"SystemExit during training: {e}")


async def try_load_existing_agent(project_folder: str) -> Optional[Agent]:
    """Try to load an existing agent from the project's models directory.

    Args:
        project_folder: Path to the project folder

    Returns:
        Loaded Agent instance if successful, None otherwise
    """
    models_dir = os.path.join(project_folder, "models")

    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        structlogger.debug("No models directory found", models_dir=models_dir)
        return None

    try:
        # Find the latest model in the models directory
        latest_model_path = get_latest_model(models_dir)
        if not latest_model_path:
            structlogger.debug(
                "No models found in models directory", models_dir=models_dir
            )
            return None

        structlogger.info(
            "Found existing model, attempting to load", model_path=latest_model_path
        )

        # Get available endpoints for agent loading
        available_endpoints = Configuration.initialise_endpoints(
            endpoints_path=Path(project_folder) / DEFAULT_ENDPOINTS_PATH
        ).endpoints
        # Get available sub agents for agent loading
        _sub_agents = Configuration.initialise_sub_agents(
            sub_agents_path=None
        ).available_agents

        # Load the agent
        agent = await load_agent(
            model_path=latest_model_path,
            endpoints=available_endpoints,
            sub_agents=_sub_agents,
        )

        if agent and agent.is_ready():
            structlogger.info(
                "Successfully loaded existing agent", model_path=latest_model_path
            )
            return agent
        else:
            structlogger.warning(
                "Agent loaded but not ready", model_path=latest_model_path
            )
            return None

    except Exception as e:
        structlogger.warning(
            "Failed to load existing agent",
            models_dir=models_dir,
            error=str(e),
            exc_info=True,
        )
        return None


async def _train_model(
    importer: TrainingDataImporter, endpoints_file: Path, config_file: Path
) -> TrainingResult:
    """Train the Rasa model."""
    try:
        structlogger.info("training.started")

        # init sub agents using the default path
        Configuration.initialise_sub_agents(sub_agents_path=None)

        training_result = await train(
            domain="",
            config=str(config_file),
            endpoints=str(endpoints_file),
            training_files=None,
            file_importer=importer,
        )

        if not training_result or not training_result.model:
            raise TrainingError("Training completed but no model was produced")

        structlogger.info("training.completed", model_path=training_result.model)

        return training_result

    except Exception as e:
        raise TrainingError(f"Model training failed: {e}")


async def _load_agent(model_path: str, endpoints_file: Path) -> Agent:
    """Load the trained agent."""
    try:
        structlogger.info("training.loading_agent", model_path=model_path)

        available_endpoints = Configuration.initialise_endpoints(
            endpoints_path=endpoints_file
        ).endpoints
        _sub_agents = Configuration.get_instance().available_agents

        if available_endpoints is None:
            raise AgentLoadError("No endpoints available for agent loading")

        structlogger.debug(
            "training.loading_agent.cwd",
            cwd=os.getcwd(),
            model_path=model_path,
        )

        agent_instance = await load_agent(
            model_path=model_path,
            remote_storage=None,
            endpoints=available_endpoints,
            sub_agents=_sub_agents,
        )

        if agent_instance is None:
            raise AgentLoadError("Agent loading returned None")

        structlogger.info("training.agent_loaded", model_path=model_path)

        return agent_instance

    except AgentLoadError:
        raise
    except Exception as e:
        raise AgentLoadError(f"Failed to load agent: {e}")
