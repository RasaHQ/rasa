from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Text

import structlog
from pydantic import Field, ValidationError, model_validator

from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    SupportedServiceType,
)
from rasa.core.redis_connection_factory import (
    DeploymentMode,
    RedisConfig,
    RedisConnectionFactory,
)
from rasa.core.timer_stores.timer_store import SessionTimer, SessionTimerStore
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.io import raise_deprecation_warning

structlogger = structlog.getLogger(__name__)

DEFAULT_SOCKET_TIMEOUT_IN_SECONDS = 10
DEFAULT_REDIS_TIMER_STORE_KEY_PREFIX = "timer:"
# Hash tag used in Cluster mode so both timer keys share the same slot
DEFAULT_REDIS_TIMER_STORE_CLUSTER_KEY_PREFIX = "{timer}:"
DEFAULT_TIMER_STORE_DB = 2

# Lua script for atomic delete (works in cluster mode unlike WATCH/MULTI/EXEC).
# KEYS[1] = timers sorted set key, KEYS[2] = data hash key
# ARGV[1] = sender_id, ARGV[2] = expected scheduled_time (or "" for unconditional)
# Returns 1 if deleted, 0 otherwise.
# For conditional delete: uses numeric comparison with small tolerance (0.001s)
# to handle floating-point representation differences between Python and Redis Lua.
DELETE_TIMER_SCRIPT = """
local zkey, dkey = KEYS[1], KEYS[2]
local sender_id, scheduled_str = ARGV[1], ARGV[2]
if scheduled_str == "" then
    local a = redis.call('ZREM', zkey, sender_id)
    local b = redis.call('HDEL', dkey, sender_id)
    return (a + b) > 0 and 1 or 0
end
local expected = tonumber(scheduled_str)
local score = redis.call('ZSCORE', zkey, sender_id)
if score == false then return 0 end
local actual = tonumber(score)
if not expected or not actual then return 0 end
if math.abs(actual - expected) > 0.001 then return 0 end
redis.call('ZREM', zkey, sender_id)
redis.call('HDEL', dkey, sender_id)
return 1
"""


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
    poll_interval: float = Field(
        default=1.0,
        description="Interval in seconds between polling for expired timers.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_url_and_host_properties(cls, data: Any) -> Any:
        """Map deprecated 'url' field to 'host' for backwards compatibility."""
        if isinstance(data, dict):
            if data.get("url") and data.get("host"):
                raise RasaException(
                    "You cannot specify both 'url' and 'host' in the Redis timer store "
                    "configuration. Please use only one of them."
                )

            if data.get("url"):
                raise_deprecation_warning(
                    "The 'url' property in the redis timer store "
                    "configuration is deprecated. Please use 'host' instead."
                )
                data["host"] = data.pop("url")
        return data


class RedisSessionTimerStore(SessionTimerStore):
    """Redis-based implementation of SessionTimerStore using sorted sets.

    In Cluster deployment_mode, the key prefix uses a Redis hash tag so both
    the timers sorted set and the data hash share the same slot (required for
    pipeline operations).
    """

    # Redis key suffixes
    TIMERS_ZSET_KEY = "timers"  # Sorted set: sender_id -> scheduled_time
    TIMER_DATA_HASH_KEY = "data"  # Hash: sender_id -> JSON timer data

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
        is_cluster = (
            getattr(config, "deployment_mode", None) == DeploymentMode.CLUSTER.value
        )
        base_prefix = (
            DEFAULT_REDIS_TIMER_STORE_CLUSTER_KEY_PREFIX
            if is_cluster
            else DEFAULT_REDIS_TIMER_STORE_KEY_PREFIX
        )
        self.key_prefix = base_prefix
        if (
            config.key_prefix
            and isinstance(config.key_prefix, str)
            and config.key_prefix.isalnum()
        ):
            structlogger.debug(
                "redis_timer_store._set_key_prefix.non_default_key_prefix",
                event_info=(
                    f"Setting non-default redis key prefix: '{config.key_prefix}'.",
                ),
            )
            self.key_prefix = config.key_prefix + ":" + base_prefix
        elif config.key_prefix:
            structlogger.warning(
                "redis_timer_store._set_key_prefix.default_instead_of_invalid_key_prefix",
                event_info=(
                    f"Omitting provided non-alphanumeric "
                    f"redis key prefix: '{config.key_prefix}'. "
                    f"Using default '{self.key_prefix}' instead."
                ),
            )
        self.red = RedisConnectionFactory.create_connection(config)
        # Register the unified Lua script for atomic delete.
        # register_script exists on both Redis and RedisCluster at runtime,
        # but the RedisCluster type stubs omit it.
        self._delete_timer_script = self.red.register_script(DELETE_TIMER_SCRIPT)  # type: ignore[union-attr]

    def _timers_key(self) -> Text:
        """Get the Redis key for the timers sorted set.

        The sorted set maps sender_id (member) -> scheduled_time (score),
        used to query expired timers by score range.
        """
        return f"{self.key_prefix}{self.TIMERS_ZSET_KEY}"

    def _data_key(self) -> Text:
        """Get the Redis key for the timer data hash.

        The hash maps sender_id (field) -> JSON-serialized timer data (value).
        """
        return f"{self.key_prefix}{self.TIMER_DATA_HASH_KEY}"

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

    async def store_timer(
        self,
        sender_id: Text,
        session_id: Optional[Text],
        scheduled_time: float,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Store timer in Redis using sorted set and hash.

        Writes to the sorted set (sender_id -> scheduled_time as score) and
        the hash (sender_id -> JSON timer data) in a single pipeline.
        In Cluster mode the key prefix uses a hash tag so both keys share a slot.
        """
        timer = SessionTimer.create_timer(
            sender_id, session_id, scheduled_time, metadata
        )

        # Pipeline (without WATCH) is safe in cluster mode because both keys
        # share the same hash slot via the {timer}: prefix in the key name.
        pipe = self.red.pipeline()
        # Add to sorted set with scheduled_time as score
        pipe.zadd(self._timers_key(), {sender_id: scheduled_time})
        # Store timer data in hash
        pipe.hset(self._data_key(), sender_id, json.dumps(timer.as_dict()))
        pipe.execute()

        structlogger.debug(
            "timer_store.redis.timer_stored",
            sender_id=sender_id,
            session_id=session_id,
            scheduled_time=scheduled_time,
        )

    async def delete_timer(
        self,
        sender_id: Text,
        only_if_scheduled_time: Optional[float] = None,
    ) -> bool:
        """Delete timer from Redis using a unified Lua script.

        If only_if_scheduled_time is None, deletes unconditionally.
        Otherwise, deletes only if the stored scheduled_time matches
        (within 0.001s tolerance) for atomic multi-pod claiming.
        """
        scheduled_str = (
            str(only_if_scheduled_time) if only_if_scheduled_time is not None else ""
        )
        deleted = bool(
            self._delete_timer_script(
                keys=[self._timers_key(), self._data_key()],
                args=[sender_id, scheduled_str],
            )
        )
        if deleted:
            structlogger.debug(
                "timer_store.redis.timer_deleted",
                sender_id=sender_id,
            )
        return deleted

    async def get_timer(self, sender_id: Text) -> Optional[SessionTimer]:
        """Get timer from Redis.

        Reads from the hash (sender_id -> JSON timer data).
        """
        data = self.red.hget(self._data_key(), sender_id)
        if data is None:
            return None

        if isinstance(data, bytes):
            data = data.decode("utf-8")
        try:
            timer_dict = json.loads(data)
            return SessionTimer.from_dict(timer_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            structlogger.warning(
                "timer_store.redis.get_timer_decode_error",
                sender_id=sender_id,
                error=str(e),
            )
            return None

    async def get_expired_timers(
        self, cutoff_time: Optional[float] = None
    ) -> List[SessionTimer]:
        """Return expired timers (for the manager to process and then delete).

        Queries the sorted set by score (scheduled_time <= cutoff_time), then
        fetches full timer data from the hash for each sender_id. Does not
        remove timers; the caller deletes them after invoking the callback.
        """
        check_time = self._check_time_for_expired(cutoff_time)

        # Get sender_ids with scores <= check_time
        expired_ids = self.red.zrangebyscore(self._timers_key(), "-inf", check_time)

        if not expired_ids:
            return []

        # Fetch timer data for expired timers
        timers = []
        for sender_id in expired_ids:
            if isinstance(sender_id, bytes):
                sender_id = sender_id.decode("utf-8")
            data = self.red.hget(self._data_key(), sender_id)
            if data:
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                try:
                    timer_dict = json.loads(data)
                    timers.append(SessionTimer.from_dict(timer_dict))
                except (json.JSONDecodeError, ValidationError) as e:
                    structlogger.warning(
                        "timer_store.redis.get_expired_timers_decode_error",
                        sender_id=sender_id,
                        error=str(e),
                    )

        return timers
