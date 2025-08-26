"""Functions for training and loading Rasa models."""

import os
import tempfile

import structlog

from rasa.builder import config
from rasa.builder.exceptions import AgentLoadError, TrainingError
from rasa.core import agent
from rasa.core.utils import AvailableEndpoints, read_endpoints_from_path
from rasa.model_training import TrainingResult, train
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string

structlogger = structlog.get_logger()


async def train_and_load_agent(importer: TrainingDataImporter) -> agent.Agent:
    """Train a model and load an agent.

    Args:
        importer: Training data importer with domain, flows, and config

    Returns:
        Loaded and ready agent

    Raises:
        TrainingError: If training fails
        AgentLoadError: If agent loading fails
    """
    try:
        # Setup endpoints for training validation
        await _setup_endpoints()

        # Train the model
        training_result = await _train_model(importer)

        # Load the agent
        agent_instance = await _load_agent(training_result.model)

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


async def _setup_endpoints() -> None:
    """Setup endpoints configuration for training."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as temp_file:
            endpoints_yaml = dump_obj_as_yaml_to_string(config.get_default_endpoints())
            temp_file.write(endpoints_yaml)
            temp_file.flush()

            # Reset and load endpoints
            AvailableEndpoints.reset_instance()
            read_endpoints_from_path(temp_file.name)

            structlogger.debug("training.endpoints_setup", temp_file=temp_file.name)

    except Exception as e:
        raise TrainingError(f"Failed to setup endpoints: {e}")


async def _train_model(importer: TrainingDataImporter) -> TrainingResult:
    """Train the Rasa model."""
    try:
        structlogger.info("training.started")

        training_result = await train(
            domain="",
            config="",
            training_files=None,
            file_importer=importer,
        )

        if not training_result or not training_result.model:
            raise TrainingError("Training completed but no model was produced")

        structlogger.info("training.completed", model_path=training_result.model)

        return training_result

    except Exception as e:
        raise TrainingError(f"Model training failed: {e}")


async def _load_agent(model_path: str) -> agent.Agent:
    """Load the trained agent."""
    try:
        structlogger.info("training.loading_agent", model_path=model_path)

        available_endpoints = AvailableEndpoints.get_instance()
        if available_endpoints is None:
            raise AgentLoadError("No endpoints available for agent loading")

        structlogger.debug(
            "training.loading_agent.cwd",
            cwd=os.getcwd(),
            model_path=model_path,
        )

        agent_instance = await agent.load_agent(
            model_path=model_path,
            remote_storage=None,
            endpoints=available_endpoints,
        )

        if agent_instance is None:
            raise AgentLoadError("Agent loading returned None")

        structlogger.info("training.agent_loaded", model_path=model_path)

        return agent_instance

    except AgentLoadError:
        raise
    except Exception as e:
        raise AgentLoadError(f"Failed to load agent: {e}")
