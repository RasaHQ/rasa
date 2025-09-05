from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError
from pytest import CaptureFixture
from redis.cluster import ClusterNode

from rasa.core.redis_connection_factory import (
    DeploymentMode,
    RedisConfig,
    RedisConnectionFactory,
)
from rasa.shared.exceptions import ConnectionException, RasaException


class TestStandardMode:
    """Tests for standard Redis deployment mode."""

    @pytest.mark.parametrize(
        "config_params,expected_redis_args",
        [
            # Default configuration
            (
                {},
                {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "username": None,
                    "password": None,
                    "ssl": False,
                    "ssl_certfile": None,
                    "ssl_keyfile": None,
                    "ssl_ca_certs": None,
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
            # Custom configuration
            (
                {
                    "host": "redis.example.com",
                    "port": 6380,
                    "db": 5,
                    "username": "testuser",
                    "password": "testpass",
                    "use_ssl": True,
                    "ssl_certfile": "/path/to/cert",
                    "socket_timeout": 30,
                    "decode_responses": True,
                },
                {
                    "host": "redis.example.com",
                    "port": 6380,
                    "db": 5,
                    "username": "testuser",
                    "password": "testpass",
                    "ssl": True,
                    "ssl_certfile": "/path/to/cert",
                    "ssl_keyfile": None,
                    "ssl_ca_certs": None,
                    "socket_timeout": 30,
                    "decode_responses": True,
                },
            ),
            # SSL configuration with all certificates
            (
                {
                    "use_ssl": True,
                    "ssl_certfile": "/path/to/cert.pem",
                    "ssl_keyfile": "/path/to/key.pem",
                    "ssl_ca_certs": "/path/to/ca.pem",
                },
                {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "username": None,
                    "password": None,
                    "ssl": True,
                    "ssl_certfile": "/path/to/cert.pem",
                    "ssl_keyfile": "/path/to/key.pem",
                    "ssl_ca_certs": "/path/to/ca.pem",
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
        ],
    )
    def test_create_connection_standard_mode(self, config_params, expected_redis_args):
        """Test standard Redis connection with various configurations."""
        with patch("redis.StrictRedis") as mock_redis:
            mock_connection = Mock()
            mock_redis.return_value = mock_connection

            config = RedisConfig(**config_params)
            result = RedisConnectionFactory.create_connection(config)

            assert result == mock_connection
            mock_redis.assert_called_once_with(**expected_redis_args)


class TestClusterMode:
    """Tests for Redis cluster deployment mode."""

    @pytest.mark.parametrize(
        "config_params,expected_hosts_ports,expected_common_args",
        [
            # With explicit endpoints
            (
                {
                    "deployment_mode": DeploymentMode.CLUSTER.value,
                    "endpoints": ["node1:6379", "node2:6380", "node3:6381"],
                    "password": "clusterpass",
                },
                [("node1", 6379), ("node2", 6380), ("node3", 6381)],
                {
                    "username": None,
                    "password": "clusterpass",
                    "ssl": False,
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
            # With endpoints, host and port
            (
                {
                    "deployment_mode": DeploymentMode.CLUSTER.value,
                    "endpoints": ["node1:6379", "node2:6380", "node3:6381"],
                    "password": "clusterpass",
                    "host": "cluster-host",
                    "port": 7000,
                },
                [("node1", 6379), ("node2", 6380), ("node3", 6381)],
                {
                    "username": None,
                    "password": "clusterpass",
                    "ssl": False,
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
            # Fallback to host/port
            (
                {
                    "deployment_mode": DeploymentMode.CLUSTER.value,
                    "host": "cluster-host",
                    "port": 7000,
                },
                [("cluster-host", 7000)],
                {
                    "username": None,
                    "password": None,
                    "ssl": False,
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
            # SSL configuration
            (
                {
                    "deployment_mode": DeploymentMode.CLUSTER.value,
                    "endpoints": ["secure-host:6380"],
                    "use_ssl": True,
                    "ssl_certfile": "/path/to/cert.pem",
                    "ssl_keyfile": "/path/to/key.pem",
                    "ssl_ca_certs": "/path/to/ca.pem",
                },
                [("secure-host", 6380)],
                {
                    "username": None,
                    "password": None,
                    "ssl": True,
                    "ssl_certfile": "/path/to/cert.pem",
                    "ssl_keyfile": "/path/to/key.pem",
                    "ssl_ca_certs": "/path/to/ca.pem",
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
        ],
    )
    def test_create_connection_cluster_mode(
        self, config_params, expected_hosts_ports, expected_common_args
    ):
        """Test cluster connection with various configurations."""
        with patch("redis.RedisCluster") as mock_cluster:
            mock_connection = Mock()
            mock_cluster.return_value = mock_connection

            config = RedisConfig(**config_params)
            result = RedisConnectionFactory.create_connection(config)

            assert result == mock_connection

            mock_cluster.assert_called_once()
            call_args = mock_cluster.call_args

            # Check that we have the right number of startup nodes
            actual_endpoints = call_args.kwargs["startup_nodes"]
            assert len(actual_endpoints) == len(expected_hosts_ports)

            # Verify each startup node is a ClusterNode with correct host and port
            for i, (expected_host, expected_port) in enumerate(expected_hosts_ports):
                cluster_node = actual_endpoints[i]
                assert isinstance(cluster_node, ClusterNode)
                assert cluster_node.host == expected_host
                assert cluster_node.port == int(expected_port)

            # Check all other parameters match
            for key, expected_value in expected_common_args.items():
                assert call_args.kwargs[key] == expected_value

    def test_create_connection_cluster_mode_db_warning(
        self,
        capsys: CaptureFixture,
    ):
        """Test cluster connection with various configurations."""
        config_params = {
            "deployment_mode": DeploymentMode.CLUSTER.value,
            "host": "cluster-host",
            "port": 7000,
            "endpoints": ["node1:6379", "node2:6380", "node3:6381"],
            "db": 5,
        }

        with patch("redis.RedisCluster") as mock_cluster:
            mock_connection = Mock()
            mock_cluster.return_value = mock_connection

            config = RedisConfig(**config_params)
            result = RedisConnectionFactory.create_connection(config)

            assert result == mock_connection
            mock_cluster.assert_called_once()
            captured = capsys.readouterr()
            assert "warning" in captured.out
            assert "Database parameter 'db=5' ignored in cluster mode" in captured.out


class TestSentinelMode:
    """Tests for Redis sentinel deployment mode."""

    @pytest.mark.parametrize(
        "config_params,expected_sentinels,expected_service,expected_sentinel_args,expected_master_args",
        [
            # With explicit endpoints and service
            (
                {
                    "deployment_mode": DeploymentMode.SENTINEL.value,
                    "endpoints": [
                        "sentinel1:26379",
                        "sentinel2:26379",
                        "sentinel3:26379",
                    ],
                    "sentinel_service": "primary",
                    "password": "sentinelpass",
                },
                [("sentinel1", 26379), ("sentinel2", 26379), ("sentinel3", 26379)],
                "primary",
                {
                    "username": None,
                    "password": "sentinelpass",
                    "socket_timeout": 10,
                },
                {
                    "db": 0,
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
            # With host, port, endpoints and service
            (
                {
                    "deployment_mode": DeploymentMode.SENTINEL.value,
                    "endpoints": [
                        "sentinel1:26379",
                        "sentinel2:26379",
                        "sentinel3:26379",
                    ],
                    "sentinel_service": "primary",
                    "password": "sentinelpass",
                    "host": "sentinel-host",
                    "port": 7000,
                },
                [("sentinel1", 26379), ("sentinel2", 26379), ("sentinel3", 26379)],
                "primary",
                {
                    "username": None,
                    "password": "sentinelpass",
                    "socket_timeout": 10,
                },
                {
                    "db": 0,
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
            # Default service name
            (
                {
                    "deployment_mode": DeploymentMode.SENTINEL.value,
                    "endpoints": ["sentinel1:26379"],
                },
                [("sentinel1", 26379)],
                "mymaster",
                {
                    "username": None,
                    "password": None,
                    "socket_timeout": 10,
                },
                {
                    "db": 0,
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
            # SSL configuration
            (
                {
                    "deployment_mode": DeploymentMode.SENTINEL.value,
                    "endpoints": ["sentinel1:26379"],
                    "use_ssl": True,
                    "ssl_certfile": "/path/to/cert.pem",
                    "ssl_keyfile": "/path/to/key.pem",
                    "ssl_ca_certs": "/path/to/ca.pem",
                },
                [("sentinel1", 26379)],
                "mymaster",
                {
                    "username": None,
                    "password": None,
                    "socket_timeout": 10,
                    "ssl": True,
                    "ssl_certfile": "/path/to/cert.pem",
                    "ssl_keyfile": "/path/to/key.pem",
                    "ssl_ca_certs": "/path/to/ca.pem",
                },
                {
                    "db": 0,
                    "socket_timeout": 10,
                    "decode_responses": False,
                },
            ),
        ],
    )
    def test_create_connection_sentinel_mode(
        self,
        config_params,
        expected_sentinels,
        expected_service,
        expected_sentinel_args,
        expected_master_args,
    ):
        """Test sentinel connection with various configurations."""
        with patch("redis.sentinel.Sentinel") as mock_sentinel_class:
            mock_sentinel = Mock()
            mock_master = Mock()
            mock_sentinel_class.return_value = mock_sentinel
            mock_sentinel.master_for.return_value = mock_master

            config = RedisConfig(**config_params)
            result = RedisConnectionFactory.create_connection(config)

            assert result == mock_master
            mock_sentinel_class.assert_called_once_with(
                expected_sentinels, **expected_sentinel_args
            )
            mock_sentinel.master_for.assert_called_once_with(
                expected_service, **expected_master_args
            )


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.parametrize(
        "config_params,expected_error_message",
        [
            (
                {"deployment_mode": "invalid_mode"},
                "Invalid deployment_mode 'invalid_mode'",
            ),
            (
                {"deployment_mode": DeploymentMode.SENTINEL.value},
                "Sentinel mode requires endpoints configuration",
            ),
            (
                {
                    "deployment_mode": DeploymentMode.CLUSTER.value,
                    "endpoints": ["invalid", "also-invalid"],
                },
                "No valid 'cluster' endpoints provided",
            ),
            (
                {
                    "deployment_mode": DeploymentMode.SENTINEL.value,
                    "endpoints": ["invalid", "also-invalid"],
                },
                "No valid 'sentinel' endpoints provided",
            ),
        ],
    )
    def test_error_scenarios(self, config_params, expected_error_message):
        """Test various error scenarios."""
        with pytest.raises(RasaException) as exc_info:
            config = RedisConfig(**config_params)
            RedisConnectionFactory.create_connection(config)

        assert expected_error_message in str(exc_info.value)


class TestEndpointParsing:
    """Tests for endpoint parsing functionality."""

    @pytest.mark.parametrize(
        "deployment_mode,endpoints,host,port,expected_result",
        [
            # Cluster mode returns dicts
            (
                DeploymentMode.CLUSTER,
                ["host1:6379", "host2:6380"],
                "localhost",
                6379,
                [("host1", 6379), ("host2", 6380)],
            ),
            # Sentinel mode returns tuples
            (
                DeploymentMode.SENTINEL,
                ["host1:26379", "host2:26380"],
                "localhost",
                6379,
                [("host1", 26379), ("host2", 26380)],
            ),
            # Standard mode returns empty list
            (
                DeploymentMode.STANDARD,
                ["host1:6379"],
                "localhost",
                6379,
                [],
            ),
        ],
    )
    def test_endpoint_parsing_different_formats(
        self, deployment_mode, endpoints, host, port, expected_result
    ):
        """Test that different deployment modes parse endpoints correctly."""
        result = RedisConnectionFactory._parse_and_validate_endpoints(
            deployment_mode,
            endpoints,
            host,
            port,
        )
        assert result == expected_result

    @pytest.mark.parametrize(
        "deployment_mode,invalid_endpoints,expected_log_message",
        [
            (
                DeploymentMode.CLUSTER,
                ["invalid-endpoint", "valid:6379", "another-invalid"],
                "Invalid format for endpoint 'invalid-endpoint'. Expected 'host:port'.",
            ),
            (
                DeploymentMode.CLUSTER,
                ["host1:invalid_port", "host2:6379"],
                (
                    "Invalid port in endpoint 'host1:invalid_port'. "
                    "Expected format 'host:port'."
                ),
            ),
            (
                DeploymentMode.CLUSTER,
                [],
                (
                    "No endpoints provided for cluster mode. "
                    "Using default 'host:port' configuration."
                ),
            ),
        ],
    )
    def test_parse_endpoints_invalid_formats(
        self,
        deployment_mode,
        invalid_endpoints,
        expected_log_message,
        capsys: CaptureFixture,
    ):
        """Test endpoint parsing with various invalid formats."""
        config_params = {
            "deployment_mode": deployment_mode.value,
            "endpoints": invalid_endpoints,
        }

        with patch("redis.RedisCluster"):
            config = RedisConfig(**config_params)
            RedisConnectionFactory.create_connection(config)
            captured = capsys.readouterr()
            assert "warning" in captured.out
            assert expected_log_message in captured.out

    @pytest.mark.parametrize(
        "invalid_config",
        [
            {"host": 123, "port": 6379},
            {"endpoints": [123, "localhost:6379"]},
        ],
    )
    def test_redis_config_validation_rejects_invalid_endpoints(self, invalid_config):
        """Test that RedisConfig properly validates endpoint types."""
        with pytest.raises(ValidationError) as exc_info:
            RedisConfig(**invalid_config)

        assert "validation error" in str(exc_info.value)

    @pytest.mark.parametrize(
        "deployment_mode,host,port,expected_result",
        [
            # Cluster mode with empty endpoints - should get default from host/port
            (
                DeploymentMode.CLUSTER,
                "redis-host",
                7000,
                ["redis-host:7000"],
            ),
            # Standard mode - should return empty regardless of input
            (
                DeploymentMode.STANDARD,
                "localhost",
                6379,
                [],
            ),
        ],
    )
    def test_get_default_endpoints(self, deployment_mode, host, port, expected_result):
        """Test default endpoint generation logic."""
        result = RedisConnectionFactory._get_default_endpoints(
            deployment_mode, host, port
        )
        assert result == expected_result

    @pytest.mark.parametrize(
        "endpoint,deployment_mode,expected_result",
        [
            # Valid cluster endpoints
            ("host1:6379", DeploymentMode.CLUSTER, ("host1", 6379)),
            (
                "redis.example.com:6380",
                DeploymentMode.CLUSTER,
                ("redis.example.com", 6380),
            ),
            # Valid sentinel endpoints
            ("sentinel1:26379", DeploymentMode.SENTINEL, ("sentinel1", 26379)),
            (
                "sentinel.example.com:26380",
                DeploymentMode.SENTINEL,
                ("sentinel.example.com", 26380),
            ),
            # Invalid formats - should return None
            ("invalid-no-port", DeploymentMode.CLUSTER, None),
            ("host:invalid_port", DeploymentMode.CLUSTER, None),
            (123, DeploymentMode.CLUSTER, None),  # Non-string type
        ],
    )
    def test_parse_single_endpoint(self, endpoint, deployment_mode, expected_result):
        """Test single endpoint parsing logic."""
        result = RedisConnectionFactory._parse_single_endpoint(
            endpoint, deployment_mode
        )
        assert result == expected_result

    @pytest.mark.parametrize(
        "endpoints,deployment_mode,expected_result",
        [
            # Valid endpoints
            (
                ["host1:6379", "host2:6380"],
                DeploymentMode.CLUSTER,
                [("host1", 6379), ("host2", 6380)],
            ),
            (
                ["sentinel1:26379", "sentinel2:26380"],
                DeploymentMode.SENTINEL,
                [("sentinel1", 26379), ("sentinel2", 26380)],
            ),
            # Mixed valid/invalid endpoints - should filter out invalid ones
            (
                ["host1:6379", "invalid-endpoint", "host2:6380"],
                DeploymentMode.CLUSTER,
                [("host1", 6379), ("host2", 6380)],
            ),
            # All invalid endpoints
            (
                ["invalid1", "invalid2"],
                DeploymentMode.CLUSTER,
                [],
            ),
            # Empty list
            (
                [],
                DeploymentMode.CLUSTER,
                [],
            ),
        ],
    )
    def test_parse_all_endpoints(self, endpoints, deployment_mode, expected_result):
        """Test parsing multiple endpoints."""
        result = RedisConnectionFactory._parse_all_endpoints(endpoints, deployment_mode)
        assert result == expected_result


class TestLogging:
    """Tests for logging functionality."""

    def test_log_cluster_db_warning_when_warning_expected(self, capsys: CaptureFixture):
        """Test that warning is logged when cluster mode uses non-zero db."""

        with patch("redis.RedisCluster"):
            RedisConnectionFactory._log_cluster_db_warning(DeploymentMode.CLUSTER, 5)
            captured = capsys.readouterr()
            assert "warning" in captured.out
            assert "Database parameter 'db=5' ignored in cluster mode" in captured.out

    @pytest.mark.parametrize(
        "deployment_mode,db",
        [
            (DeploymentMode.STANDARD, 5),
            (DeploymentMode.CLUSTER, 0),
        ],
    )
    def test_log_cluster_db_warning_when_no_warning_expected(
        self, deployment_mode, db, capsys: CaptureFixture
    ):
        """Test that no warning is logged when appropriate."""
        with patch("redis.RedisCluster"):
            RedisConnectionFactory._log_cluster_db_warning(deployment_mode, db)
            captured = capsys.readouterr()
            assert "warning" not in captured.out
            assert (
                "Database parameter 'db=5' ignored in cluster mode" not in captured.out
            )

    def test_cluster_initialization_logging(self, capsys: CaptureFixture):
        """Test that cluster initialization is logged."""
        config_params = {
            "deployment_mode": DeploymentMode.CLUSTER.value,
            "endpoints": ["host1:6379", "host2:6379"],
        }

        with patch("redis.RedisCluster"):
            config = RedisConfig(**config_params)
            RedisConnectionFactory.create_connection(config)

            captured = capsys.readouterr()
            assert "info" in captured.out
            assert "Initializing Redis Cluster" in captured.out

    def test_sentinel_initialization_logging(self, capsys: CaptureFixture):
        """Test that sentinel initialization is logged."""
        config_params = {
            "deployment_mode": DeploymentMode.SENTINEL.value,
            "endpoints": ["sentinel1:26379", "sentinel2:26380"],
            "sentinel_service": "mymaster",
        }

        with patch("redis.sentinel.Sentinel"):
            config = RedisConfig(**config_params)
            RedisConnectionFactory.create_connection(config)

            captured = capsys.readouterr()
            assert "info" in captured.out
            assert "Initializing Redis Sentinel" in captured.out

    def test_standard_initialization_logging(self, capsys: CaptureFixture):
        """Test that standard initialization is logged."""
        config_params = {
            "deployment_mode": DeploymentMode.STANDARD.value,
            "endpoints": ["host1:6379"],
        }

        with patch("redis.StrictRedis"):
            config = RedisConfig(**config_params)
            RedisConnectionFactory.create_connection(config)

            captured = capsys.readouterr()
            assert "info" in captured.out
            assert "Initializing Redis connection" in captured.out


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_cluster_mode_with_single_valid_endpoint_from_mixed(self):
        """Test cluster mode filters out invalid endpoints and uses valid ones."""
        # Given
        config_params = {
            "deployment_mode": DeploymentMode.CLUSTER.value,
            "endpoints": ["host1:invalid_port", "host2:6379"],
        }

        with patch("redis.RedisCluster") as mock_cluster:
            # When
            config = RedisConfig(**config_params)
            RedisConnectionFactory.create_connection(config)

            # Then
            call_args = mock_cluster.call_args
            assert len(call_args.kwargs["startup_nodes"]) == 1

            cluster_node = call_args.kwargs["startup_nodes"][0]
            assert isinstance(cluster_node, ClusterNode)
            assert cluster_node.host == "host2"
            assert cluster_node.port == 6379

    def test_sentinel_mode_with_single_valid_endpoint_from_mixed(self):
        """Test sentinel mode filters out invalid endpoints and uses valid ones."""
        # Given
        config_params = {
            "deployment_mode": DeploymentMode.SENTINEL.value,
            "endpoints": ["invalid-endpoint", "sentinel1:26379"],
            "sentinel_service": "custom",
        }

        with patch("redis.sentinel.Sentinel") as mock_sentinel_class:
            mock_sentinel = Mock()
            mock_sentinel_class.return_value = mock_sentinel

            # When
            config = RedisConfig(**config_params)
            RedisConnectionFactory.create_connection(config)

            # Then
            call_args = mock_sentinel_class.call_args
            assert len(call_args.args[0]) == 1
            assert call_args.args[0][0] == ("sentinel1", 26379)


class TestConnectionExceptionHandling:
    """Tests for connection exception handling in all modes."""

    def test_standard_connection_exception_handling(self):
        """Test that standard connection failures are properly handled."""
        config_params = {
            "deployment_mode": DeploymentMode.STANDARD.value,
            "endpoints": ["host1:0000"],
        }

        with patch("redis.StrictRedis") as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")

            with pytest.raises(ConnectionException) as exc_info:
                config = RedisConfig(**config_params)
                RedisConnectionFactory.create_connection(config)

            assert "Error initializing Redis connection" in str(exc_info.value)
            assert "Connection failed" in str(exc_info.value)

    def test_cluster_connection_exception_handling(self):
        """Test that cluster connection failures are properly handled."""
        config_params = {
            "deployment_mode": DeploymentMode.CLUSTER.value,
            "endpoints": ["host1:0000"],
        }

        with patch("redis.RedisCluster") as mock_cluster:
            mock_cluster.side_effect = Exception("Connection failed")

            with pytest.raises(ConnectionException) as exc_info:
                config = RedisConfig(**config_params)
                RedisConnectionFactory.create_connection(config)

            assert "Error initializing Redis Cluster" in str(exc_info.value)
            assert "Connection failed" in str(exc_info.value)

    def test_sentinel_connection_exception_handling(self):
        """Test that Sentinel connection failures are properly handled."""
        config_params = {
            "deployment_mode": DeploymentMode.SENTINEL.value,
            "endpoints": ["host1:0000"],
        }

        with patch("redis.sentinel.Sentinel") as mock_sentinel:
            mock_master = Mock()
            mock_master.ping.side_effect = Exception("Connection failed")
            mock_sentinel.return_value.master_for.return_value = mock_master

            with pytest.raises(ConnectionException) as exc_info:
                config = RedisConfig(**config_params)
                RedisConnectionFactory.create_connection(config)

            assert "Error initializing Redis Sentinel" in str(exc_info.value)
            assert "Connection failed" in str(exc_info.value)
