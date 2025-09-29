import os
from enum import Enum
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import redis
import structlog
from pydantic import BaseModel, ConfigDict

from rasa.core.constants import (
    AWS_ELASTICACHE_CLUSTER_NAME_ENV_VAR_NAME,
    REDIS_SERVICE_NAME,
)
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProvider,
    IAMCredentialsProviderInput,
    SupportedServiceType,
    create_iam_credentials_provider,
)
from rasa.shared.exceptions import ConnectionException, RasaException

structlogger = structlog.getLogger(__name__)

DEFAULT_SOCKET_TIMEOUT_IN_SECONDS = 10


class DeploymentMode(Enum):
    """Supported Redis deployment modes."""

    STANDARD = "standard"
    CLUSTER = "cluster"
    SENTINEL = "sentinel"


class StandardRedisConfig(BaseModel):
    """Base configuration for Redis connections."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    host: Text = "localhost"
    port: int = 6379
    username: Optional[Text] = None
    password: Optional[Text] = None
    use_ssl: bool = False
    ssl_keyfile: Optional[Text] = None
    ssl_certfile: Optional[Text] = None
    ssl_ca_certs: Optional[Text] = None
    db: int = 0
    socket_timeout: float = DEFAULT_SOCKET_TIMEOUT_IN_SECONDS
    decode_responses: bool = False
    iam_credentials_provider: Optional[IAMCredentialsProvider] = None


class ClusterRedisConfig(StandardRedisConfig):
    """Configuration for Redis Cluster connections."""

    endpoints: List[Tuple[Text, int]]


class SentinelRedisConfig(StandardRedisConfig):
    """Configuration for Redis Sentinel connections."""

    endpoints: List[Tuple[Text, int]]
    sentinel_service: Optional[Text] = "mymaster"


class RedisConfig(BaseModel):
    """Base configuration for Redis connections."""

    host: Text = "localhost"
    port: int = 6379
    service_type: SupportedServiceType
    username: Optional[Text] = None
    password: Optional[Text] = None
    use_ssl: bool = False
    ssl_keyfile: Optional[Text] = None
    ssl_certfile: Optional[Text] = None
    ssl_ca_certs: Optional[Text] = None
    db: int = 0
    socket_timeout: float = DEFAULT_SOCKET_TIMEOUT_IN_SECONDS
    decode_responses: bool = False
    deployment_mode: Text = DeploymentMode.STANDARD.value
    endpoints: Optional[List[Text]] = None
    sentinel_service: Optional[Text] = None


class RedisConnectionFactory:
    """Factory class for creating Redis connections with different modes."""

    @classmethod
    def create_connection(
        cls,
        config: RedisConfig,
    ) -> Union[redis.Redis, redis.RedisCluster]:
        """Create a Redis connection based on the configuration.

        Args:
            config: Redis configuration object containing all connection parameters.

        Returns:
            A Redis connection - either a standard Redis connection, a RedisCluster
            connection, or a Redis master connection managed by Sentinel.

        Raises:
            RasaException: If configuration is invalid.
        """
        if config.endpoints is None:
            config.endpoints = []

        try:
            deployment_mode_enum = DeploymentMode(config.deployment_mode)
        except ValueError:
            valid_modes = [mode.value for mode in DeploymentMode]
            raise RasaException(
                f"Invalid deployment_mode '{config.deployment_mode}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )

        parsed_endpoints = cls._parse_and_validate_endpoints(
            deployment_mode_enum, config.endpoints, config.host, config.port
        )

        iam_credentials_provider = create_iam_credentials_provider(
            IAMCredentialsProviderInput(
                service_type=config.service_type,
                service_name=REDIS_SERVICE_NAME,
                username=config.username,
                cluster_name=os.getenv(AWS_ELASTICACHE_CLUSTER_NAME_ENV_VAR_NAME),
            )
        )

        if deployment_mode_enum == DeploymentMode.CLUSTER:
            cls._log_cluster_db_warning(deployment_mode_enum, config.db)
            cluster_config = ClusterRedisConfig(
                username=config.username,
                password=config.password,
                use_ssl=config.use_ssl,
                ssl_certfile=config.ssl_certfile,
                ssl_keyfile=config.ssl_keyfile,
                ssl_ca_certs=config.ssl_ca_certs,
                db=config.db,
                socket_timeout=config.socket_timeout,
                decode_responses=config.decode_responses,
                endpoints=parsed_endpoints,
                iam_credentials_provider=iam_credentials_provider,
            )
            return cls._create_cluster_connection(cluster_config)
        elif deployment_mode_enum == DeploymentMode.SENTINEL:
            sentinel_kwargs = {
                "username": config.username,
                "password": config.password,
                "use_ssl": config.use_ssl,
                "ssl_certfile": config.ssl_certfile,
                "ssl_keyfile": config.ssl_keyfile,
                "ssl_ca_certs": config.ssl_ca_certs,
                "db": config.db,
                "socket_timeout": config.socket_timeout,
                "decode_responses": config.decode_responses,
                "endpoints": parsed_endpoints,
                "iam_credentials_provider": iam_credentials_provider,
            }

            if config.sentinel_service is not None:
                sentinel_kwargs["sentinel_service"] = config.sentinel_service

            sentinel_config = SentinelRedisConfig(**sentinel_kwargs)
            return cls._create_sentinel_connection(sentinel_config)
        else:
            standard_config = StandardRedisConfig(
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
                use_ssl=config.use_ssl,
                ssl_certfile=config.ssl_certfile,
                ssl_keyfile=config.ssl_keyfile,
                ssl_ca_certs=config.ssl_ca_certs,
                db=config.db,
                socket_timeout=config.socket_timeout,
                decode_responses=config.decode_responses,
                iam_credentials_provider=iam_credentials_provider,
            )
            return cls._create_standard_connection(standard_config)

    @classmethod
    def _parse_and_validate_endpoints(
        cls,
        deployment_mode: DeploymentMode,
        endpoints: List[str],
        host: Text,
        port: int,
    ) -> List[Union[Dict[str, Any], Tuple[str, int]]]:
        """Parse and validate endpoints based on deployment mode."""
        if deployment_mode == DeploymentMode.STANDARD:
            if endpoints:
                structlogger.warning(
                    "redis_connection_factory.standard_endpoints_ignored",
                    event_info="Parameter `endpoints` ignored in standard mode. "
                    "Only 'host' and 'port' are used for standard Redis connections.",
                )
            return []

        if not endpoints:
            endpoints = cls._get_default_endpoints(deployment_mode, host, port)

        # Parse endpoints into appropriate format
        parsed_endpoints = cls._parse_all_endpoints(endpoints, deployment_mode)

        if not parsed_endpoints:
            raise RasaException(
                f"No valid '{deployment_mode.value}' endpoints provided"
            )

        return parsed_endpoints

    @classmethod
    def _get_default_endpoints(
        cls,
        deployment_mode: DeploymentMode,
        host: Text,
        port: int,
    ) -> List[str]:
        """Get default endpoints when none provided."""
        if deployment_mode == DeploymentMode.CLUSTER:
            structlogger.warning(
                "redis_connection_factory.cluster_endpoints_not_provided",
                event_info="No endpoints provided for cluster mode. "
                "Using default 'host:port' configuration.",
            )
            return [f"{host}:{port}"]
        elif deployment_mode == DeploymentMode.SENTINEL:
            raise RasaException("Sentinel mode requires endpoints configuration")

        return []

    @classmethod
    def _parse_single_endpoint(
        cls, endpoint: str, deployment_mode: DeploymentMode
    ) -> Optional[Union[Dict[str, Any], Tuple[str, int]]]:
        """Parse a single endpoint string into the appropriate format."""
        if not isinstance(endpoint, str):
            structlogger.warning(
                f"redis_connection_factory.invalid_{deployment_mode.value}_endpoint_type",
                event_info=f"Invalid endpoint type for endpoint '{endpoint}'. "
                "Expected string in 'host:port' format.",
            )
            return None

        if ":" not in endpoint:
            structlogger.warning(
                f"redis_connection_factory.invalid_{deployment_mode.value}_endpoint_format",
                event_info=f"Invalid format for endpoint '{endpoint}'. "
                "Expected 'host:port'.",
            )
            return None

        host, port_str = endpoint.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            structlogger.warning(
                f"redis_connection_factory.invalid_{deployment_mode.value}_endpoint",
                event_info=f"Invalid port in endpoint '{endpoint}'. "
                "Expected format 'host:port'.",
            )
            return None

        return (host, port)

    @classmethod
    def _parse_all_endpoints(
        cls,
        endpoints: List[str],
        deployment_mode: DeploymentMode,
    ) -> List[Union[Dict[str, Any], Tuple[str, int]]]:
        """Parse a list of endpoint strings into appropriate format."""
        parsed_endpoints = []
        for endpoint in endpoints:
            parsed = cls._parse_single_endpoint(endpoint, deployment_mode)
            if parsed:
                parsed_endpoints.append(parsed)
        return parsed_endpoints

    @classmethod
    def _create_cluster_connection(
        cls,
        config: ClusterRedisConfig,
    ) -> redis.RedisCluster:
        """Create a Redis Cluster connection.

        Note: Database parameter is ignored in cluster mode (always uses db=0).

        Args:
            config: Cluster configuration containing all connection parameters.

        Returns:
            redis.RedisCluster: Configured cluster connection.

        Raises:
            ConnectionException: If cluster initialization fails.
        """
        from redis.cluster import ClusterNode

        structlogger.info(
            "redis_connection_factory.cluster_mode",
            event_info=f"Initializing Redis Cluster with {len(config.endpoints)} nodes",
        )

        cluster_nodes = [ClusterNode(host, port) for host, port in config.endpoints]

        common_config_kwargs = {
            "startup_nodes": cluster_nodes,
            "ssl": config.use_ssl,
            "ssl_certfile": config.ssl_certfile,
            "ssl_keyfile": config.ssl_keyfile,
            "ssl_ca_certs": config.ssl_ca_certs,
            "socket_timeout": config.socket_timeout,
            "decode_responses": config.decode_responses,
        }

        try:
            if config.iam_credentials_provider is not None:
                structlogger.debug("redis_connection_factory.cluster_iam_auth_enabled")

                redis_cluster: redis.RedisCluster = redis.RedisCluster(
                    credential_provider=config.iam_credentials_provider,
                    **common_config_kwargs,
                )
            else:
                redis_cluster = redis.RedisCluster(
                    username=config.username,
                    password=config.password,
                    **common_config_kwargs,
                )
        except Exception as e:
            raise ConnectionException(f"Error initializing Redis Cluster: {e}")

        return redis_cluster

    @classmethod
    def _create_sentinel_connection(
        cls,
        config: SentinelRedisConfig,
    ) -> redis.Redis:
        """Create a Sentinel-managed Redis connection.

        Connects to Redis master through Sentinel service discovery.
        Tests connection with ping() before returning.

        Args:
            config: Sentinel configuration containing all connection parameters.

        Returns:
            redis.Redis: Connection to the Redis master via sentinel.

        Raises:
            ConnectionException: If sentinel initialization or connection test fails.
        """
        from redis.sentinel import Sentinel

        structlogger.info(
            "redis_connection_factory.sentinel_mode",
            event_info=f"Initializing Redis Sentinel with {len(config.endpoints)} "
            f"sentinel endpoints and service: {config.sentinel_service}",
        )

        # Configuration for Sentinel connection
        connection_kwargs: Dict[str, Any] = {
            "socket_timeout": config.socket_timeout,
        }

        sentinel_kwargs: Optional[Dict] = None
        if config.iam_credentials_provider is not None:
            structlogger.debug("redis_connection_factory.sentinel_iam_auth_enabled")
            sentinel_kwargs = {"credential_provider": config.iam_credentials_provider}
        else:
            connection_kwargs.update(
                {
                    "username": config.username,
                    "password": config.password,
                }
            )

        # SSL configuration
        if config.use_ssl:
            connection_kwargs.update(
                {
                    "ssl": config.use_ssl,
                    "ssl_certfile": config.ssl_certfile,
                    "ssl_keyfile": config.ssl_keyfile,
                    "ssl_ca_certs": config.ssl_ca_certs,
                }
            )

        # Configuration for Redis client (master/replica)
        client_kwargs = {
            "db": config.db,
            "decode_responses": config.decode_responses,
            "socket_timeout": config.socket_timeout,
        }

        # Create Sentinel instance
        try:
            sentinel = Sentinel(
                config.endpoints, sentinel_kwargs=sentinel_kwargs, **connection_kwargs
            )
            master = sentinel.master_for(config.sentinel_service, **client_kwargs)

            # Test the connection
            master.ping()

        except Exception as e:
            raise ConnectionException(f"Error initializing Redis Sentinel: {e}")

        return master

    @classmethod
    def _create_standard_connection(
        cls,
        config: StandardRedisConfig,
    ) -> redis.Redis:
        """Create a standard Redis connection.

        Args:
            config: Standard configuration containing all connection parameters.

        Returns:
            redis.Redis: Configured Redis connection.
        """
        structlogger.info(
            "redis_connection_factory.standard_mode",
            event_info="Initializing Redis connection",
        )
        # Build connection arguments
        connection_args: Dict[str, Any] = {
            "host": config.host,
            "port": int(config.port),
            "db": config.db,
            "socket_timeout": float(config.socket_timeout),
            "ssl": config.use_ssl,
            "ssl_certfile": config.ssl_certfile,
            "ssl_keyfile": config.ssl_keyfile,
            "ssl_ca_certs": config.ssl_ca_certs,
            "decode_responses": config.decode_responses,
        }

        if config.iam_credentials_provider is not None:
            structlogger.debug("redis_connection_factory.standard_iam_auth_enabled")
            connection_args.update(
                {"credential_provider": config.iam_credentials_provider}
            )
        else:
            connection_args.update(
                {
                    "password": config.password,
                    "username": config.username,
                }
            )

        try:
            standard_redis = redis.StrictRedis(**connection_args)
        except Exception as e:
            raise ConnectionException(f"Error initializing Redis connection: {e}")

        return standard_redis

    @classmethod
    def _log_cluster_db_warning(cls, deployment_mode: DeploymentMode, db: int) -> None:
        """Log warning if db parameter is set in cluster mode."""
        if deployment_mode == DeploymentMode.CLUSTER and db != 0:
            structlogger.warning(
                "redis_connection_factory.cluster_db_ignored",
                event_info=f"Database parameter 'db={db}' ignored in cluster mode. "
                "Redis Cluster only supports database 0.",
            )
