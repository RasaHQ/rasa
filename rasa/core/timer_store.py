from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Literal, Optional, Text, Union

import structlog
from pydantic import Field

import rasa.shared.utils.common
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    SupportedServiceType,
)
from rasa.core.redis_connection_factory import (
    RedisConfig,
    RedisConnectionFactory,
)
from rasa.shared.exceptions import ConnectionException
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.getLogger(__name__)

DEFAULT_SOCKET_TIMEOUT_IN_SECONDS = 10
DEFAULT_REDIS_TIMER_STORE_KEY_PREFIX = "timer:"
DEFAULT_TIMER_STORE_DB = 2


class SessionTimerStore:
    """Base class for timer stores."""

    @staticmethod
    def create(
        obj: Union[SessionTimerStore, EndpointConfig, None],
    ) -> SessionTimerStore:
        """Factory to create a timer store."""
        if isinstance(obj, SessionTimerStore):
            return obj

        try:
            return _create_from_endpoint_config(obj)
        except ConnectionError as error:
            raise ConnectionException("Cannot connect to timer store.") from error

    @abstractmethod
    def close(self) -> None:
        """Close the timer store connection.

        Subclasses must implement this method to handle any necessary cleanup.
        """
        raise NotImplementedError()


class RedisSessionTimerStoreConfig(RedisConfig):
    """Configuration for Redis-based timer store."""

    type: Literal["redis"] = "redis"
    service_type: SupportedServiceType = SupportedServiceType.TIMER_STORE
    db: int = DEFAULT_TIMER_STORE_DB
    key_prefix: Optional[str] = Field(
        default=None,
        description="Namespace prefix to prepend to timer keys. Must be alphanumeric.",
    )
    socket_timeout: float = Field(
        default=DEFAULT_SOCKET_TIMEOUT_IN_SECONDS,
        description="Time in seconds after which Redis commands will time out.",
    )


class RedisSessionTimerStore(SessionTimerStore):
    """Redis-based implementation of SessionTimerStore."""

    def __init__(
        self,
        config: Optional[RedisSessionTimerStoreConfig] = None,
    ) -> None:
        """Initialize the Redis timer store.

        Args:
            config: Configuration for the Redis connection.
        """
        if config is None:
            config = RedisSessionTimerStoreConfig()

        self.config = config
        self.key_prefix = DEFAULT_REDIS_TIMER_STORE_KEY_PREFIX
        if config.key_prefix:
            structlogger.debug(
                "redis_timer_store._set_key_prefix.non_default_key_prefix",
                event_info=(
                    f"Setting non-default redis key prefix: '{config.key_prefix}'.",
                ),
            )
            self._set_key_prefix(config.key_prefix)
        self.red = RedisConnectionFactory.create_connection(config)

    def _set_key_prefix(self, key_prefix: Text) -> None:
        """Set the key prefix, validating it is alphanumeric.

        Args:
            key_prefix: The namespace prefix to prepend to the default prefix.
        """
        if isinstance(key_prefix, str) and key_prefix.isalnum():
            self.key_prefix = key_prefix + ":" + DEFAULT_REDIS_TIMER_STORE_KEY_PREFIX
        else:
            structlogger.warning(
                "redis_timer_store._set_key_prefix.default_instead_of_invalid_key_prefix",
                event_info=(
                    f"Omitting provided non-alphanumeric "
                    f"redis key prefix: '{key_prefix}'. "
                    f"Using default '{self.key_prefix}' instead."
                ),
            )

    def close(self) -> None:
        """Close the Redis connection."""
        if hasattr(self, "red") and self.red is not None:
            try:
                self.red.close()
            except Exception as e:
                structlogger.warning(
                    "timer_store.redis.connection_close_failed",
                    error=str(e),
                )


class InMemorySessionTimerStore(SessionTimerStore):
    """In-memory store for timers."""

    def __init__(self) -> None:
        """Initialize the in-memory timer store."""
        self.timers: Dict[Text, Any] = {}
        super().__init__()

    def close(self) -> None:
        """No cleanup needed for in-memory store."""
        pass


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None,
) -> SessionTimerStore:
    """Given an endpoint configuration, create a proper `SessionTimerStore` object."""
    if (
        endpoint_config is None
        or endpoint_config.type is None
        or endpoint_config.type == "in_memory"
    ):
        timer_store: SessionTimerStore = InMemorySessionTimerStore()
    elif endpoint_config.type == "redis":
        config = RedisSessionTimerStoreConfig.model_validate(endpoint_config.to_dict())
        timer_store = RedisSessionTimerStore(config)
    else:
        timer_store = _load_from_module_name_in_endpoint_config(endpoint_config)

    structlogger.debug(
        "timer_store._create_from_endpoint_config.timer_store_connected",
        event_info=f"Connected to timer store '{timer_store.__class__.__name__}'.",
    )

    return timer_store


def _load_from_module_name_in_endpoint_config(
    endpoint_config: EndpointConfig,
) -> SessionTimerStore:
    """Retrieve a `SessionTimerStore` based on its class name."""
    try:
        timer_store_class = rasa.shared.utils.common.class_from_module_path(
            endpoint_config.type
        )
        return timer_store_class(endpoint_config=endpoint_config)
    except (AttributeError, ImportError) as e:
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a `SessionTimerStore` "
            f"instance. Error: {e}"
        )
