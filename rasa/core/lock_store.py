from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Text, Union

import structlog
from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    NonNegativeInt,
    ValidationError,
    model_validator,
)

import rasa.shared.utils.common
from rasa.core.constants import DEFAULT_LOCK_LIFETIME, IAM_CLOUD_PROVIDER_ENV_VAR_NAME
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    SupportedServiceType,
)
from rasa.core.lock import TicketLock
from rasa.core.redis_connection_factory import (
    DeploymentMode,
    RedisConfig,
    RedisConnectionFactory,
)
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.shared.utils.io import raise_deprecation_warning
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.getLogger(__name__)


def _get_lock_lifetime() -> int:
    return int(os.environ.get("TICKET_LOCK_LIFETIME", 0)) or DEFAULT_LOCK_LIFETIME


LOCK_LIFETIME = _get_lock_lifetime()
DEFAULT_SOCKET_TIMEOUT_IN_SECONDS = 10

DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX = "lock:"


# noinspection PyUnresolvedReferences
class LockError(RasaException):
    """Exception that is raised when a lock cannot be acquired.

    Attributes:
         message (str): explanation of which `conversation_id` raised the error
    """

    pass


class LockStore:
    """Base class for ticket locks."""

    @staticmethod
    def create(obj: Union[LockStore, EndpointConfig, None]) -> LockStore:
        """Factory to create a lock store."""
        if isinstance(obj, LockStore):
            return obj

        try:
            return _create_from_endpoint_config(obj)
        except ConnectionError as error:
            raise ConnectionException("Cannot connect to lock store.") from error

    @staticmethod
    def create_lock(conversation_id: Text) -> TicketLock:
        """Create a new `TicketLock` for `conversation_id`."""
        return TicketLock(conversation_id)

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """Fetch lock for `conversation_id` from storage."""
        raise NotImplementedError

    def delete_lock(self, conversation_id: Text) -> None:
        """Delete lock for `conversation_id` from storage."""
        raise NotImplementedError

    def save_lock(self, lock: TicketLock) -> None:
        """Commit `lock` to storage."""
        raise NotImplementedError

    def issue_ticket(
        self, conversation_id: Text, lock_lifetime: float = LOCK_LIFETIME
    ) -> int:
        """Issue new ticket with `lock_lifetime` for lock associated with
        `conversation_id`.

        Creates a new lock if none is found.
        """
        structlogger.debug(
            "lock_store.issue_ticket",
            event_info=f"Issuing ticket for conversation '{conversation_id}'.",
        )
        try:
            lock = self.get_or_create_lock(conversation_id)
            ticket = lock.issue_ticket(lock_lifetime)
            self.save_lock(lock)

            return ticket
        except Exception as e:
            raise LockError(f"Error while acquiring lock. Error:\n{e}")

    @asynccontextmanager
    async def lock(
        self,
        conversation_id: Text,
        lock_lifetime: float = LOCK_LIFETIME,
        wait_time_in_seconds: float = 1,
    ) -> AsyncGenerator[TicketLock, None]:
        """Acquire lock with lifetime `lock_lifetime`for `conversation_id`.

        Try acquiring lock with a wait time of `wait_time_in_seconds` seconds
        between attempts. Raise a `LockError` if lock has expired.
        """
        ticket = self.issue_ticket(conversation_id, lock_lifetime)
        try:
            yield await self._acquire_lock(
                conversation_id, ticket, wait_time_in_seconds
            )
        finally:
            self.cleanup(conversation_id, ticket)

    async def _acquire_lock(
        self, conversation_id: Text, ticket: int, wait_time_in_seconds: float
    ) -> TicketLock:
        structlogger.debug(
            "lock_store._acquiring_lock_for_conversation",
            event_info=f"Acquiring lock for conversation '{conversation_id}'.",
        )
        while True:
            # fetch lock in every iteration because lock might no longer exist
            lock = self.get_lock(conversation_id)

            # exit loop if lock does not exist anymore (expired)
            if not lock:
                break

            # acquire lock if it isn't locked
            if not lock.is_locked(ticket):
                structlogger.debug(
                    "lock_store._acquired_lock_for_conversation",
                    event_info=f"Acquired lock for conversation '{conversation_id}'.",
                )
                return lock

            items_before_this = ticket - (lock.now_serving or 0)

            structlogger.debug(
                "lock_store._retrying_lock_acquisition",
                event_info=(
                    f"Failed to acquire lock for conversation ID '{conversation_id}' "
                    f"because {items_before_this} other item(s) for this "
                    f"conversation ID have to be finished processing first. "
                    f"Retrying in {wait_time_in_seconds} seconds ..."
                ),
            )

            # sleep and update lock
            await asyncio.sleep(wait_time_in_seconds)
            self.update_lock(conversation_id)

        raise LockError(
            f"Could not acquire lock for conversation_id '{conversation_id}'."
        )

    def update_lock(self, conversation_id: Text) -> None:
        """Fetch lock for `conversation_id`, remove expired tickets and save lock."""
        lock = self.get_lock(conversation_id)
        if lock:
            lock.remove_expired_tickets()
            self.save_lock(lock)

    def get_or_create_lock(self, conversation_id: Text) -> TicketLock:
        """Fetch existing lock for `conversation_id`.

        Alternatively, create a new one if it doesn't exist.
        """
        existing_lock = self.get_lock(conversation_id)

        if existing_lock:
            return existing_lock

        return self.create_lock(conversation_id)

    def is_someone_waiting(self, conversation_id: Text) -> bool:
        """Return whether someone is waiting for lock for this `conversation_id`."""
        lock = self.get_lock(conversation_id)
        if lock:
            return lock.is_someone_waiting()

        return False

    def finish_serving(self, conversation_id: Text, ticket_number: int) -> None:
        """Finish serving ticket with `ticket_number` for `conversation_id`.

        Removes ticket from lock and saves lock.
        """
        lock = self.get_lock(conversation_id)
        if lock:
            lock.remove_ticket_for(ticket_number)
            self.save_lock(lock)

    def cleanup(self, conversation_id: Text, ticket_number: int) -> None:
        """Remove lock for `conversation_id` if no one is waiting."""
        self.finish_serving(conversation_id, ticket_number)
        if not self.is_someone_waiting(conversation_id):
            self.delete_lock(conversation_id)

    @staticmethod
    def _log_deletion(conversation_id: Text, deletion_successful: bool) -> None:
        if deletion_successful:
            structlogger.debug(
                "lock_store._deleted_lock_for_conversation",
                event_info=f"Deleted lock for conversation '{conversation_id}'.",
            )
        else:
            structlogger.debug(
                "lock_store._failed_to_delete_lock_for_conversation",
                event_info=(
                    f"Could not delete lock for conversation '{conversation_id}'."
                ),
            )


class RedisLockStoreConfig(BaseModel):
    host: Union[AnyUrl, Literal["localhost"], str] = Field(
        default="localhost", description="The host of the redis server."
    )
    port: NonNegativeInt = Field(
        default=6379, ge=0, le=65535, description="The port of the redis server."
    )
    db: NonNegativeInt = Field(
        default=0,
        ge=0,
        description="The name of the database within Redis "
        "which should be used by Rasa",
    )
    username: Optional[str] = Field(
        default=None,
        description="The username which should be used for "
        "authentication with the Redis database.",
    )
    password: Optional[str] = Field(
        default=None,
        description="The username which should be used for "
        "authentication with the Redis database.",
    )
    use_ssl: bool = Field(
        default=False,
        serialization_alias="ssl",
        description="True if SSL should be used for the connection to Redis.",
    )
    ssl_certfile: Optional[str] = Field(
        default=None,
        description="Path to the SSL certificate file.",
    )
    ssl_keyfile: Optional[str] = Field(
        default=None, description="Path to the SSL private key file."
    )
    ssl_ca_certs: Optional[str] = Field(
        default=None, description="Path to the SSL CA certificate file."
    )
    key_prefix: Optional[str] = Field(
        default=None,
        description="Prefix to prepend to all keys "
        "used by the lock store. Must be alphanumeric.",
    )
    socket_timeout: float = Field(
        default=DEFAULT_SOCKET_TIMEOUT_IN_SECONDS,
        description="Timeout in seconds after which an exception "
        "will be raised in case Redis doesn't respond "
        "within `socket_timeout` seconds.",
    )
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.STANDARD,
        description="Redis deployment mode: 'standard', 'cluster', or 'sentinel'",
    )
    endpoints: Optional[List[str]] = Field(
        default=None,
        description="List of endpoints for cluster/sentinel mode in 'host:port' format",
    )
    sentinel_service: Optional[str] = Field(
        default=None,
        description="Sentinel service name",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_url_and_host_properties(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if bool(data.get("url", None)) and bool(data.get("host", None)):
                raise RasaException(
                    "You cannot specify both 'url' and 'host' in the Redis lock store "
                    "configuration. Please use only one of them."
                )

            if data.get("url", None):
                raise_deprecation_warning(
                    "The 'url' property in the redis lock store "
                    "configuration is deprecated. Please use 'host' instead."
                )
                data["host"] = data.pop("url")
        return data

    @model_validator(mode="after")
    def verify_username_password(self) -> RedisLockStoreConfig:
        if os.getenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME) is None and (
            bool(self.username) ^ bool(self.password)
        ):
            raise ValueError(
                f"Expected username and password. "
                f"Found: username: {'<has value>' if self.username else '<N/A>'}, "
                f"password: {'<has value>' if self.password else '<N/A>'}"
            )
        return self


class RedisLockStore(LockStore):
    """Redis store for ticket locks."""

    def __init__(
        self,
        config: RedisLockStoreConfig = RedisLockStoreConfig(),
    ) -> None:
        """Create a lock store which uses Redis for persistence.

        Args:
            config: Redis lock store configuration.
        """
        self.config = config
        try:
            redis_config = RedisConfig(
                host=str(self.config.host),
                port=self.config.port,
                service_type=SupportedServiceType.LOCK_STORE,
                db=self.config.db,
                username=self.config.username,
                password=self.config.password,
                use_ssl=self.config.use_ssl,
                ssl_keyfile=self.config.ssl_keyfile,
                ssl_certfile=self.config.ssl_certfile,
                ssl_ca_certs=self.config.ssl_ca_certs,
                deployment_mode=self.config.deployment_mode.value,
                endpoints=self.config.endpoints,
                sentinel_service=self.config.sentinel_service,
                socket_timeout=self.config.socket_timeout,
            )
            self.red = RedisConnectionFactory.create_connection(redis_config)
        except ValidationError as e:
            raise RasaException(f"Invalid Redis configuration: {e}")

        self.key_prefix = DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX
        if self.config.key_prefix:
            structlogger.debug(
                "redis_lock_store._set_key_prefix.non_default_key_prefix",
                event_info=(
                    f"Setting non-default "
                    f"redis key prefix: '{self.config.key_prefix}'.",
                ),
            )
            self._set_key_prefix(self.config.key_prefix)

        super().__init__()

    def _set_key_prefix(self, key_prefix: Text) -> None:
        if isinstance(key_prefix, str) and key_prefix.isalnum():
            self.key_prefix = key_prefix + ":" + DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX
        else:
            structlogger.warning(
                "redis_lock_store._set_key_prefix.default_instead_of_invalid_key_prefix",
                event_info=(
                    f"Omitting provided non-alphanumeric "
                    f"redis key prefix: '{key_prefix}'. "
                    f"Using default '{self.key_prefix}' instead."
                ),
            )

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """Retrieves lock (see parent docstring for more information)."""
        serialised_lock = self.red.get(self.key_prefix + conversation_id)
        if serialised_lock:
            # Handle bytes to string conversion for JSON parsing
            if isinstance(serialised_lock, bytes):
                serialised_lock = serialised_lock.decode("utf-8")
            return TicketLock.from_dict(json.loads(serialised_lock))

        return None

    def delete_lock(self, conversation_id: Text) -> None:
        """Deletes lock for conversation ID."""
        deletion_successful = self.red.delete(self.key_prefix + conversation_id)
        self._log_deletion(conversation_id, deletion_successful)

    def save_lock(self, lock: TicketLock) -> None:
        self.red.set(self.key_prefix + lock.conversation_id, lock.dumps())


class InMemoryLockStore(LockStore):
    """In-memory store for ticket locks."""

    def __init__(self) -> None:
        """Initialise dictionary of locks."""
        self.conversation_locks: Dict[Text, TicketLock] = {}
        super().__init__()

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """Get lock for conversation if it exists."""
        return self.conversation_locks.get(conversation_id)

    def delete_lock(self, conversation_id: Text) -> None:
        """Delete lock for conversation."""
        deleted_lock = self.conversation_locks.pop(conversation_id, None)
        self._log_deletion(
            conversation_id, deletion_successful=deleted_lock is not None
        )

    def save_lock(self, lock: TicketLock) -> None:
        """Save lock in store."""
        self.conversation_locks[lock.conversation_id] = lock


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None,
) -> LockStore:
    """Given an endpoint configuration, create a proper `LockStore` object."""
    if (
        endpoint_config is None
        or endpoint_config.type is None
        or endpoint_config.type == "in_memory"
    ):
        # this is the default type if no lock store type is set

        lock_store: LockStore = InMemoryLockStore()
    elif endpoint_config.type == "redis":
        config = RedisLockStoreConfig.model_validate(endpoint_config.to_dict())

        lock_store = RedisLockStore(config)
    elif endpoint_config.type == "concurrent_redis":
        from rasa.core.concurrent_lock_store import ConcurrentRedisLockStore

        lock_store = ConcurrentRedisLockStore(endpoint_config=endpoint_config)
    else:
        lock_store = _load_from_module_name_in_endpoint_config(endpoint_config)

    structlogger.debug(
        "lock_store._create_from_endpoint_config.lock_store_connected",
        event_info=f"Connected to lock store '{lock_store.__class__.__name__}'.",
    )

    return lock_store


def _load_from_module_name_in_endpoint_config(
    endpoint_config: EndpointConfig,
) -> LockStore:
    """Retrieve a `LockStore` based on its class name."""
    try:
        lock_store_class = rasa.shared.utils.common.class_from_module_path(
            endpoint_config.type
        )
        return lock_store_class(endpoint_config=endpoint_config)
    except (AttributeError, ImportError) as e:
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a `LockStore` "
            f"instance. Error: {e}"
        )
