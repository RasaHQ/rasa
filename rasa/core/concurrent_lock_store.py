import json
import time
from collections import deque
from typing import Deque, Optional, Text

import structlog
from pydantic import ValidationError

from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    SupportedServiceType,
)
from rasa.core.lock import Ticket, TicketLock
from rasa.core.lock_store import (
    DEFAULT_SOCKET_TIMEOUT_IN_SECONDS,
    LOCK_LIFETIME,
    LockError,
    LockStore,
)
from rasa.core.redis_connection_factory import (
    DeploymentMode,
    RedisConfig,
    RedisConnectionFactory,
)
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

DEFAULT_REDIS_DB = 1

DEFAULT_PORT = 6379

DEFAULT_HOSTNAME = "localhost"

structlogger = structlog.getLogger(__name__)

LAST_ISSUED_TICKET_NUMBER_SUFFIX = "last_issued_ticket_number"

DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX = "concurrent_lock:"


class ConcurrentTicketLock(TicketLock):
    def concurrent_issue_ticket(self, lifetime: float, ticket_number: int) -> int:
        """Issue a new ticket and return its number.

        Args:
            lifetime: The length of time after which the ticket expires.
            ticket_number: Integer representing the ticket number.

        Returns:
            ticket_number integer.

        """
        ticket = Ticket(ticket_number, time.time() + lifetime)
        self.tickets.append(ticket)

        return ticket_number


class ConcurrentRedisLockStore(LockStore):
    """Concurrent implementation of a Redis store for ticket locks."""

    def __init__(
        self,
        endpoint_config: EndpointConfig,
    ) -> None:
        """Create a lock store which uses Redis for persistence.

        Args:
            endpoint_config: The endpointConfig defined the lock store.

            Can contain the following properties:
            host - The host of the redis server.
            port - The port of the redis server.
            db - The name of the database within Redis which should be used by Rasa
                Open Source.
            username - The username which should be used for authentication with the
                Redis database.
            password - The password which should be used for authentication with the
                Redis database.
            use_ssl - `True` if SSL should be used for the connection to Redis.
            ssl_certfile - Path to the SSL certificate file.
            ssl_keyfile - Path to the SSL private key file.
            ssl_ca_certs - Path to the SSL CA certificate file.
            key_prefix - prefix to prepend to all keys used by the lock store. Must be
                alphanumeric.
            socket_timeout - Timeout in seconds after which an exception will be raised
                in case Redis doesn't respond within `socket_timeout` seconds.
            deployment_mode - Redis deployment mode: standard, cluster, or sentinel.
            endpoints - List of endpoints for cluster/sentinel mode in host:port format.
            sentinel_service - Sentinel service name.
        """
        host = endpoint_config.kwargs.get("host", DEFAULT_HOSTNAME)
        port = endpoint_config.kwargs.get("port", DEFAULT_PORT)
        db = endpoint_config.kwargs.get("db", DEFAULT_REDIS_DB)
        username = endpoint_config.kwargs.get("username")
        password = endpoint_config.kwargs.get("password")
        use_ssl = bool(endpoint_config.kwargs.get("use_ssl", False))
        ssl_certfile = endpoint_config.kwargs.get("ssl_certfile")
        ssl_keyfile = endpoint_config.kwargs.get("ssl_keyfile")
        ssl_ca_certs = endpoint_config.kwargs.get("ssl_ca_certs")
        key_prefix = endpoint_config.kwargs.get("key_prefix")
        socket_timeout = endpoint_config.kwargs.get(
            "socket_timeout", DEFAULT_SOCKET_TIMEOUT_IN_SECONDS
        )
        deployment_mode = endpoint_config.kwargs.get(
            "deployment_mode", DeploymentMode.STANDARD.value
        )
        endpoints = endpoint_config.kwargs.get("endpoints", [])
        sentinel_service = endpoint_config.kwargs.get("sentinel_service")

        try:
            redis_config = RedisConfig(
                host=host,
                port=port,
                service_type=SupportedServiceType.LOCK_STORE,
                db=db,
                username=username,
                password=password,
                use_ssl=use_ssl,
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile,
                ssl_ca_certs=ssl_ca_certs,
                socket_timeout=socket_timeout,
                deployment_mode=deployment_mode,
                endpoints=endpoints,
                sentinel_service=sentinel_service,
            )
            self.red = RedisConnectionFactory.create_connection(redis_config)
        except ValidationError as e:
            raise RasaException(f"Invalid Redis configuration: {e}")

        self.deployment_mode = deployment_mode
        self.key_prefix = DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX
        if key_prefix:
            structlogger.debug(
                "concurrent_redis_lock_store._set_key_prefix.non_default_key_prefix",
                event_info=f"Setting non-default redis key prefix: '{key_prefix}'.",
            )
            self._set_key_prefix(key_prefix)

        super().__init__()

    def _set_key_prefix(self, key_prefix: Text) -> None:
        if isinstance(key_prefix, str) and key_prefix.isalnum():
            self.key_prefix = (
                key_prefix + ":" + DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX
            )
        else:
            structlogger.warning(
                "concurrent_redis_lock_store._set_key_prefix.default_instead_of_invalid_key_prefix",
                event_info=(
                    f"Omitting provided non-alphanumeric "
                    f"redis key prefix: '{key_prefix}'. "
                    f"Using default '{self.key_prefix}' instead."
                ),
            )

    def _scan_cluster_keys(self, pattern: Text) -> list:
        """Scan keys in cluster mode with proper cursor handling."""
        keys = []
        cursor = 0

        while True:
            try:
                cursor, batch_keys = self.red.scan(cursor, match=pattern, count=100)
                keys.extend(batch_keys)

                if isinstance(cursor, dict):
                    # cursor is a dict mapping each node to its scan position. e.g
                    # {'127.0.0.1:7000': 0, '127.0.0.1:7001': 5, '127.0.0.1:7002': 0}
                    # A cursor value of 0 means that node has finished scanning
                    # When all nodes show 0, the entire cluster scan is complete
                    if all(v == 0 for v in cursor.values()):
                        break
                else:
                    # if scan is complete
                    if cursor == 0:
                        break

            except Exception as e:
                structlogger.warning(
                    "concurrent_redis_lock_store._get_keys_by_pattern.scan_interrupted",
                    event_info=f"SCAN interrupted in cluster mode: {e}. "
                    f"Returning {len(keys)} keys found so far.",
                )
                break

        return keys

    def _get_keys_by_pattern(self, pattern: Text) -> list:
        """Get keys by pattern, using SCAN for cluster mode and KEYS for others."""
        if self.deployment_mode == DeploymentMode.CLUSTER.value:
            return self._scan_cluster_keys(pattern)
        else:
            return self.red.keys(pattern)

    def issue_ticket(
        self, conversation_id: Text, lock_lifetime: float = LOCK_LIFETIME
    ) -> int:
        """Issue new ticket for a conversation.

        It's configured with `lock_lifetime` and associated with `conversation_id`.
        Creates a new lock if none is found.
        """
        structlogger.debug(
            "concurrent_redis_lock_store.issue_ticket",
            event_info=f"Issuing ticket for conversation '{conversation_id}'.",
        )
        try:
            lock = self.get_or_create_lock(conversation_id)
            lock.remove_expired_tickets()
            ticket_number = self.increment_ticket_number(lock)
            ticket = lock.concurrent_issue_ticket(lock_lifetime, ticket_number)  # type: ignore[attr-defined]
            self.save_lock(lock)

            return ticket
        except Exception as e:
            raise LockError(f"Error while acquiring lock. Error:\n{e}")

    def get_lock(self, conversation_id: Text) -> Optional[ConcurrentTicketLock]:
        """Retrieves lock (see parent docstring for more information)."""
        tickets: Deque[Ticket] = deque()

        pattern = self.key_prefix + conversation_id + ":" + "[0-9]*"
        redis_keys = self._get_keys_by_pattern(pattern)

        for key in redis_keys:
            serialised_ticket = self.red.get(key)
            if serialised_ticket:
                # Handle bytes to string conversion for JSON parsing
                if isinstance(serialised_ticket, bytes):
                    serialised_ticket = serialised_ticket.decode("utf-8")
                ticket = Ticket.from_dict(json.loads(serialised_ticket))
                tickets.appendleft(ticket)

        tickets = deque(sorted(tickets, key=lambda x: x.number))

        return ConcurrentTicketLock(conversation_id, tickets)

    def delete_lock(self, conversation_id: Text) -> None:
        """Deletes lock for conversation ID."""
        pattern = self.key_prefix + conversation_id + ":*"
        redis_keys = self._get_keys_by_pattern(pattern)

        if not redis_keys:
            structlogger.debug(
                "concurrent_redis_lock_store.delete_lock_key_not_found",
                event_info=(
                    f"The lock store does not contain any key-value "
                    f"items for conversation '{conversation_id}'."
                ),
            )
            return None

        deletion_successful = self.red.delete(*redis_keys)
        if deletion_successful == 0:
            self._log_deletion(conversation_id, False)
        else:
            self._log_deletion(conversation_id, True)

    def save_lock(self, lock: TicketLock) -> None:
        """Commit individual tickets and last issued ticket number to storage."""
        last_issued_ticket = lock.tickets[-1]
        serialised_ticket = last_issued_ticket.dumps()
        key = (
            self.key_prefix
            + lock.conversation_id
            + ":"
            + str(last_issued_ticket.number)
        )
        self.red.set(
            name=key, value=serialised_ticket, ex=int(last_issued_ticket.expires)
        )

    def increment_ticket_number(self, lock: TicketLock) -> int:
        """Uses Redis atomic transaction to increment ticket number."""
        last_issued_key = (
            self.key_prefix
            + lock.conversation_id
            + ":"
            + LAST_ISSUED_TICKET_NUMBER_SUFFIX
        )

        return self.red.incr(name=last_issued_key)

    def finish_serving(self, conversation_id: Text, ticket_number: int) -> None:
        """Finish serving ticket with `ticket_number` for `conversation_id`.

        Removes ticket from storage.
        """
        ticket_key = self.key_prefix + conversation_id + ":" + str(ticket_number)
        self.red.delete(ticket_key)
