from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, Optional

import structlog

from rasa.cli.validation.config_path_validation import get_validated_path
from rasa.core.available_agents import DEFAULT_AGENTS_CONFIG_FOLDER, AvailableAgents
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.credentials import CredentialsConfig
from rasa.core.config.message_procesing_config import MessageProcessingConfig
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_ENDPOINTS_PATH,
)

logger = structlog.get_logger(__name__)


@dataclasses.dataclass
class ValidatedConfigPaths:
    credentials_path: Optional[Path] = None
    endpoints_path: Optional[Path] = None
    message_processing_config_path: Optional[Path] = None


@dataclasses.dataclass
class ConfigPath:
    target: Optional[Path] = None
    type: Literal["credentials", "endpoints", "config"] = "endpoints"
    alternative_targets: list[Path] = dataclasses.field(default_factory=list)
    default_path: str = ""

    @staticmethod
    def _validate(
        config_path: ConfigPath, can_be_missing: bool = True
    ) -> Optional[Path]:
        validated_path = get_validated_path(
            config_path.target,
            config_path.type,
            config_path.alternative_targets,
            can_be_missing,
        )

        if validated_path is None and can_be_missing:
            logger.debug(
                f"No {config_path.type} configuration file found. "
                f"Proceeding without {config_path.type} configuration."
            )

        if isinstance(validated_path, str):
            validated_path = Path(validated_path)

        return validated_path


@dataclasses.dataclass
class CredentialsConfigPath(ConfigPath):
    def __init__(self, target: Optional[Path] = None):
        super().__init__(
            target=target,
            type="credentials",
            alternative_targets=[Path(DEFAULT_CREDENTIALS_PATH)],
            default_path=DEFAULT_CREDENTIALS_PATH,
        )

    @staticmethod
    def validate(
        target_config_path: Optional[Path] = None, can_be_missing: bool = True
    ) -> Optional[Path]:
        config_path = CredentialsConfigPath(target=target_config_path)
        return ConfigPath._validate(config_path, can_be_missing)

    @staticmethod
    def default_file_path() -> Path:
        return Path(DEFAULT_CREDENTIALS_PATH)

    @staticmethod
    def at_path(
        root_path: Path,
    ) -> Path:
        return root_path / CredentialsConfigPath.default_file_path()


@dataclasses.dataclass
class EndpointsConfigPath(ConfigPath):
    def __init__(self, target: Optional[Path] = None):
        super().__init__(
            target=target,
            type="endpoints",
            alternative_targets=[Path(DEFAULT_ENDPOINTS_PATH)],
            default_path=DEFAULT_ENDPOINTS_PATH,
        )

    @staticmethod
    def validate(
        target_config_path: Optional[Path] = None, can_be_missing: bool = True
    ) -> Optional[Path]:
        config_path = EndpointsConfigPath(target=target_config_path)
        return ConfigPath._validate(config_path, can_be_missing)

    @staticmethod
    def default_file_path() -> Path:
        return Path(DEFAULT_ENDPOINTS_PATH)

    @staticmethod
    def at_path(
        root_path: Path,
    ) -> Path:
        return root_path / EndpointsConfigPath.default_file_path()


@dataclasses.dataclass
class MessageProcessingConfigPath(ConfigPath):
    def __init__(self, target: Optional[Path] = None):
        super().__init__(
            target=target,
            type="config",
            alternative_targets=[Path(DEFAULT_CONFIG_PATH)],
            default_path=DEFAULT_CONFIG_PATH,
        )

    @staticmethod
    def validate(
        target_config_path: Optional[Path] = None, can_be_missing: bool = True
    ) -> Optional[Path]:
        config_path = MessageProcessingConfigPath(target=target_config_path)
        return ConfigPath._validate(config_path, can_be_missing)

    @staticmethod
    def default_file_path() -> Path:
        return Path(DEFAULT_CONFIG_PATH)

    @staticmethod
    def at_path(
        root_path: Path,
    ) -> Path:
        return root_path / MessageProcessingConfigPath.default_file_path()


class Configuration:
    _instance: Optional[Configuration] = None

    def __init__(
        self,
        endpoints: AvailableEndpoints,
        available_agents: AvailableAgents,
        credentials: Optional[CredentialsConfig] = None,
        message_processing_config: Optional[MessageProcessingConfig] = None,
    ):
        self.credentials = credentials
        self.available_agents = available_agents
        self.endpoints = endpoints
        self.message_processing_config = message_processing_config

    @classmethod
    def initialise_empty(cls) -> Configuration:
        if cls._instance is None:
            cls._instance = Configuration(
                credentials=None,
                endpoints=AvailableEndpoints(),
                available_agents=AvailableAgents(),
                message_processing_config=None,
            )
        return cls._instance

    @classmethod
    def initialise_endpoints(cls, endpoints_path: Path) -> Configuration:
        logger.debug(
            "configuration.initialise_endpoints.start", endpoints_path=endpoints_path
        )

        validated_endpoints_path = EndpointsConfigPath.validate(endpoints_path)
        endpoints = (
            AvailableEndpoints.read_endpoints(validated_endpoints_path)
            if validated_endpoints_path
            else AvailableEndpoints()
        )
        if cls._instance is None:
            cls._instance = Configuration(
                endpoints=endpoints,
                available_agents=AvailableAgents(),
                credentials=None,
                message_processing_config=None,
            )
        else:
            cls._instance.endpoints = endpoints
        return cls._instance

    @classmethod
    def initialise_empty_endpoints(cls) -> Configuration:
        logger.debug("configuration.initialise_empty_endpoints.start")
        if cls._instance is None:
            cls._instance = Configuration(
                endpoints=AvailableEndpoints(),
                available_agents=AvailableAgents(),
                credentials=None,
                message_processing_config=None,
            )
        else:
            cls._instance.endpoints = AvailableEndpoints()
        return cls._instance

    @classmethod
    def initialise_message_processing(
        cls, message_processing_config_path: Path
    ) -> Configuration:
        logger.debug(
            "configuration.initialise_message_processing.start",
            message_processing_config_path=message_processing_config_path,
        )
        validated_message_processing_path = MessageProcessingConfigPath.validate(
            message_processing_config_path
        )
        message_processing_config = (
            MessageProcessingConfig.load_from_file(validated_message_processing_path)
            if validated_message_processing_path
            else None
        )
        if cls._instance is None:
            cls._instance = Configuration(
                message_processing_config=message_processing_config,
                credentials=None,
                endpoints=AvailableEndpoints(),
                available_agents=AvailableAgents(),
            )
        else:
            cls._instance.message_processing_config = message_processing_config
        return cls._instance

    @classmethod
    def initialise_credentials(cls, credentials_path: Path) -> Configuration:
        logger.debug(
            "configuration.initialise_credentials.start",
            credentials_path=credentials_path,
        )

        validated_credentials_path = CredentialsConfigPath.validate(credentials_path)

        credentials = (
            CredentialsConfig.load_from_file(validated_credentials_path)
            if validated_credentials_path
            else None
        )
        if cls._instance is None:
            cls._instance = Configuration(
                credentials=credentials,
                endpoints=AvailableEndpoints(),
                available_agents=AvailableAgents(),
                message_processing_config=None,
            )
        else:
            cls._instance.credentials = credentials
        return cls._instance

    @classmethod
    def initialise_sub_agents(cls, sub_agents_path: Optional[Path]) -> Configuration:
        sub_agents_folder = (
            str(sub_agents_path)
            if sub_agents_path is not None
            else DEFAULT_AGENTS_CONFIG_FOLDER
        )
        logger.debug(
            "configuration.initialise_sub_agents.start",
            sub_agents_folder=sub_agents_folder,
        )
        available_agents = AvailableAgents.read_from_folder(sub_agents_folder)

        if cls._instance is None:
            cls._instance = Configuration(
                endpoints=AvailableEndpoints(),
                available_agents=available_agents,
                credentials=None,
                message_processing_config=None,
            )
        else:
            cls._instance.available_agents = available_agents
        return cls._instance

    @classmethod
    def get_instance(cls) -> Configuration:
        if cls._instance is None:
            raise Exception(
                "Configuration not initialized. "
                "Call appropriate 'initialise' methods to "
                "load the config when Rasa Pro starts: "
                "initialise_endpoints(), "
                "initialise_sub_agents(), "
                "initialise_credentials(), "
                "initialise_message_processing()"
            )
        return cls._instance
