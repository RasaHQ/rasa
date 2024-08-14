import base64
import datetime
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Text
from unittest.mock import MagicMock, call

import pytest
from apscheduler.triggers.interval import IntervalTrigger
from pytest import LogCaptureFixture, MonkeyPatch

from rasa.core.secrets_manager.endpoints import EndpointTrait
from rasa.core.secrets_manager.vault import (
    VaultCredentialsLocation,
    VaultEndpointConfigReader,
    VaultSecretsManager,
    VaultTokenManager,
)
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

# file deepcode ignore NoHardcodedPasswords/test: Test credentials


@pytest.fixture
def mock_hvac_client(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_hvac_client = MagicMock()
    monkeypatch.setattr("hvac.Client", _mock_hvac_client)
    return _mock_hvac_client


@pytest.fixture
def mock_datetime(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_datetime = MagicMock()
    monkeypatch.setattr("rasa.core.secrets_manager.vault.datetime", _mock_datetime)
    return _mock_datetime


@pytest.fixture
def mock_background_scheduler(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_background_scheduler = MagicMock()
    monkeypatch.setattr(
        "rasa.core.secrets_manager.vault.BackgroundScheduler",
        _mock_background_scheduler,
    )
    return _mock_background_scheduler


@pytest.fixture
def mock_token_manager(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_token_manager = MagicMock()
    monkeypatch.setattr(
        "rasa.core.secrets_manager.vault.VaultTokenManager", _mock_token_manager
    )
    return _mock_token_manager


@pytest.fixture
def mock_tracker_store_validator(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_tracker_store_validator = MagicMock()
    monkeypatch.setattr(
        "rasa.core.secrets_manager.vault.TrackerStoreEndpointValidator",
        _mock_tracker_store_validator,
    )
    return _mock_tracker_store_validator


@pytest.fixture
def mock_get_vault_endpoint_reader(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_get_vault_endpoint_reader = MagicMock()
    monkeypatch.setattr(
        "rasa.core.secrets_manager.vault.get_vault_endpoint_reader",
        _mock_get_vault_endpoint_reader,
    )
    return _mock_get_vault_endpoint_reader


def test_vault_token_manager_initialization(mock_hvac_client: MagicMock) -> None:
    # This test is not really necessary, but it's here to make sure that the
    # VaultTokenManager can be initialized with the default values.
    host = "some_host"
    token = "some_token"
    client = {"url": host, "token": token}
    mock_hvac_client.return_value = client
    token_manager = VaultTokenManager(host=host, token=token)

    assert token_manager.client == client
    assert token_manager.host == host
    assert token_manager.token == token


def test_vault_token_manager_start_with_renewable_token(
    mock_hvac_client: MagicMock, monkeypatch: MonkeyPatch
) -> None:
    # This test is not really necessary, but it's here to make sure that the
    # VaultTokenManager can be initialized with the default values.
    host = "some_host"
    token = "some_token"
    client = MagicMock()
    client.auth.token.lookup_self.return_value = {
        "data": {"renewable": True, "ttl": 100, "creation_ttl": 100}
    }
    mock_hvac_client.return_value = client
    token_manager = VaultTokenManager(host=host, token=token)

    mock_start_token_refresh = MagicMock()
    monkeypatch.setattr(token_manager, "_start_token_refresh", mock_start_token_refresh)
    token_manager.start()
    mock_start_token_refresh.assert_called_once_with(100, 100)


def test_vault_token_manager_start_with_non_renewable_token(
    mock_hvac_client: MagicMock, monkeypatch: MonkeyPatch
) -> None:
    # This test is not really necessary, but it's here to make sure that the
    # VaultTokenManager can be initialized with the default values.
    host = "some_host"
    token = "some_token"
    client = MagicMock()
    client.auth.token.lookup_self.return_value = {"data": {"renewable": False}}
    mock_hvac_client.return_value = client
    token_manager = VaultTokenManager(host=host, token=token)

    mock_start_token_refresh = MagicMock()
    monkeypatch.setattr(token_manager, "_start_token_refresh", mock_start_token_refresh)

    token_manager.start()
    mock_start_token_refresh.assert_not_called()


@pytest.mark.parametrize(
    "remaining_life_in_seconds, ttl_in_seconds, expected_start_offset_in_seconds, expected_interval_in_seconds",  # noqa: E501
    [
        (100, 100, 85, 85),
        (100, 200, 85, 185),
        # when remaining_life_in_seconds is less than default
        # TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS
        (
            4,
            50,
            VaultTokenManager.TOKEN_MINIMAL_REMAINING_LIFE_IN_SECONDS,
            35,
        ),
        # when ttl_in_seconds is less than default
        # TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS
        (
            100,
            4,
            85,
            VaultTokenManager.TOKEN_MINIMAL_REFRESH_TIME_IN_SECONDS,
        ),
    ],
)
def test_vault_token_manager__start_token_refresh(
    mock_background_scheduler: MagicMock,
    mock_datetime: MagicMock,
    remaining_life_in_seconds: int,
    ttl_in_seconds: int,
    expected_start_offset_in_seconds: int,
    expected_interval_in_seconds: int,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_datetime.now = MagicMock()
    fixed_date_time = datetime.datetime(2010, 1, 1)
    mock_datetime.now.return_value = fixed_date_time
    background_scheduler_instance = mock_background_scheduler.return_value
    background_scheduler_instance.start = MagicMock()
    background_scheduler_instance.add_job = MagicMock()

    token_manager = VaultTokenManager(host="", token="")
    monkeypatch.setattr(token_manager, "_renew_token", MagicMock())
    token_manager._start_token_refresh(remaining_life_in_seconds, ttl_in_seconds)

    interval = IntervalTrigger(
        seconds=expected_interval_in_seconds,
        start_date=fixed_date_time
        + datetime.timedelta(seconds=expected_start_offset_in_seconds),
    )
    trigger = background_scheduler_instance.add_job.call_args[1]["trigger"]
    assert trigger.interval == interval.interval
    assert trigger.start_date == interval.start_date


@pytest.mark.usefixtures("mock_hvac_client")
def test_vault_token_manager_renew_token(
    monkeypatch: MonkeyPatch,
) -> None:
    token_manager = VaultTokenManager(host="", token="")

    mock_renew_self = MagicMock()
    monkeypatch.setattr(token_manager.client.auth.token, "renew_self", mock_renew_self)
    token_manager._renew_token()


@pytest.mark.usefixtures("mock_hvac_client")
def test_vault_token_manager_renew_token_fails_once_and_succeeds_afterwards(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    token_manager = VaultTokenManager(host="", token="", number_of_retries=2)

    mock_renew_self = MagicMock()
    exception = Exception("some error")
    mock_renew_self.side_effect = [exception, None]
    monkeypatch.setattr(token_manager.client.auth.token, "renew_self", mock_renew_self)
    time_to_wait_between_retries = 0
    monkeypatch.setattr(
        token_manager,
        "TIME_TO_WAIT_BETWEEN_RETRIES_IN_SECONDS",
        time_to_wait_between_retries,
    )
    token_manager._renew_token()

    log_msg = (
        f"Failed to renew vault token. Error: {exception}. "
        f"Trying again in {time_to_wait_between_retries} second"
    )
    assert log_msg in caplog.text


@pytest.mark.usefixtures("mock_hvac_client")
def test_vault_token_manager_renew_token_fails_to_renew(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    token_manager = VaultTokenManager(host="", token="", number_of_retries=1)

    mock_renew_self = MagicMock()
    exception = Exception("some error")
    mock_renew_self.side_effect = [exception, None]
    monkeypatch.setattr(token_manager.client.auth.token, "renew_self", mock_renew_self)
    monkeypatch.setattr(token_manager, "TIME_TO_WAIT_BETWEEN_RETRIES_IN_SECONDS", 0)

    with pytest.raises(RasaException):
        token_manager._renew_token()


@pytest.mark.parametrize(
    "original_secrets, transit_keys, decrypted_passwords, expected_secrets",
    [
        (
            dict(
                {
                    "username": "vault:v1:some_username",
                    "password": "vault:v1:some_key",
                }
            ),
            dict({"username": "some_transit_key", "password": "some_transit_key"}),
            ["some decrypted username", "some_decrypted_password"],
            {
                "username": "some decrypted username",
                "password": "some_decrypted_password",
            },
        ),
        (
            dict(
                {
                    "username": "vault:v1:some_username",
                    "password": "vault:v1:some_key",
                }
            ),
            dict({"password": "some_transit_key"}),
            ["some_decrypted_password"],
            {
                "username": "vault:v1:some_username",
                "password": "some_decrypted_password",
            },
        ),
        (
            dict(
                {
                    "username": "vault:v1:some_username",
                    "password": "vault:v1:some_key",
                }
            ),
            dict(),
            [],
            {
                "username": "vault:v1:some_username",
                "password": "vault:v1:some_key",
            },
        ),
    ],
)
@pytest.mark.usefixtures("mock_hvac_client")
def test_vault_secret_manager_decode_transit_secrets(
    mock_token_manager: MagicMock,
    original_secrets: Dict[Text, Text],
    transit_keys: Dict[Text, Text],
    decrypted_passwords: List[Text],
    expected_secrets: Dict[Text, Text],
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    token_manager_instance = mock_token_manager.return_value
    token_manager_instance.start = MagicMock()

    vault = VaultSecretsManager(host="", token="", secrets_path="")

    secrets = deepcopy(original_secrets)

    mock_decrypt_transit_cipher = MagicMock()
    mock_decrypt_transit_cipher.side_effect = decrypted_passwords
    monkeypatch.setattr(vault, "_decrypt_transit_cipher", mock_decrypt_transit_cipher)

    with caplog.at_level(logging.INFO):
        vault._decode_transit_secrets(secrets=secrets, transit_keys=transit_keys)

    mock_decrypt_transit_cipher.assert_has_calls(
        [
            call(
                cipher=original_secrets[key],
                transit_key_name=transit_keys[key],
            )
            for key in transit_keys.keys()
        ]
    )

    assert secrets == expected_secrets

    start_decrypting_secret_log_message = (
        "Start to decrypt secrets using transit secrets engine."
    )
    assert start_decrypting_secret_log_message in caplog.text
    finished_decrypting_secret_log_message = (
        "Finished decrypting secrets using transit secrets engine."
    )
    assert finished_decrypting_secret_log_message in caplog.text


def test_vault_secret_manager_decrypt_transit_cipher(
    mock_hvac_client: MagicMock,
    mock_token_manager: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    token_manager_instance = mock_token_manager.return_value
    token_manager_instance.start = MagicMock()

    decrypted_password = "some_decrypted_password"
    encoded_password = base64.standard_b64encode(decrypted_password.encode("utf-8"))

    client = MagicMock()
    client.secrets.transit.decrypt_data.return_value = dict(
        {"data": dict({"plaintext": encoded_password})}
    )
    mock_hvac_client.return_value = client

    transit_key = "some_transit_key"
    ciphertext = "some_encrypted_password"
    transit_mount_point = "transit"
    vault = VaultSecretsManager(
        host="", token="", secrets_path="", transit_mount_point=transit_mount_point
    )

    with caplog.at_level(logging.INFO):
        result = vault._decrypt_transit_cipher(
            cipher=ciphertext, transit_key_name=transit_key
        )

    client.secrets.transit.decrypt_data.assert_called_once_with(
        name=transit_key,
        ciphertext=ciphertext,
        mount_point=transit_mount_point,
    )
    assert result == decrypted_password
    start_decrypting_log_message = "Decrypting cipher using transit key."
    assert start_decrypting_log_message in caplog.text
    finished_decrypting_log_message = "Finished decrypting cipher using transit key."
    assert finished_decrypting_log_message in caplog.text


@pytest.mark.parametrize(
    "cipher, expected",
    [
        ("vault:v1:some_encrypted_password", True),
        ("vault:v2:some_encrypted_password", True),
        (":vault:v2:some_encrypted_password", False),
        (":v2:some_encrypted_password", False),
        ("v2:some_encrypted_password", False),
        ("vault:some_encrypted_password", False),
        ("some_encrypted_password", False),
    ],
)
def test_vault_secret_manager_is_vault_transit_cipher(
    cipher: Text, expected: bool
) -> None:
    assert VaultSecretsManager._is_vault_transit_cipher(cipher=cipher) == expected


def test_vault_secret_manager_load_secrets_without_transit_keys(
    mock_hvac_client: MagicMock,
    mock_token_manager: MagicMock,
    mock_tracker_store_validator: MagicMock,
    mock_get_vault_endpoint_reader: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    token_manager_instance = mock_token_manager.return_value
    token_manager_instance.start = MagicMock()

    vault_credentials_processor_instance = mock_tracker_store_validator.return_value
    vault_credentials_processor_instance.get_transit_keys_per_endpoint_property = (
        MagicMock()
    )
    vault_credentials_processor_instance.get_transit_keys_per_endpoint_property.return_value = (  # noqa: E501
        None
    )

    endpoint_trait = EndpointTrait(
        endpoint_type="some_endpoint", endpoint_config=EndpointConfig()
    )
    mock_get_vault_endpoint_reader.return_value = vault_credentials_processor_instance

    client = MagicMock()
    secrets = dict(
        {
            "username": "some_username",
            "password": "some_password",
        }
    )
    client.secrets.kv.read_secret_version.return_value = dict(
        {"data": dict({"data": secrets})}
    )
    mock_hvac_client.return_value = client

    vault = VaultSecretsManager(host="", token="", secrets_path="")

    mock_decode_transit_secrets = MagicMock()
    monkeypatch.setattr(vault, "_decode_transit_secrets", mock_decode_transit_secrets)

    result = vault.load_secrets(endpoint_trait=endpoint_trait)
    assert result == secrets

    mock_decode_transit_secrets.assert_not_called()
    mock_get_vault_endpoint_reader.assert_called_once_with(
        endpoint_trait=endpoint_trait
    )
    vault_credentials_processor_instance.get_transit_keys_per_endpoint_property.assert_called_once()


def test_vault_secret_manager_load_secrets_with_transit_keys(
    mock_hvac_client: MagicMock,
    mock_token_manager: MagicMock,
    mock_tracker_store_validator: MagicMock,
    mock_get_vault_endpoint_reader: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    token_manager_instance = mock_token_manager.return_value
    token_manager_instance.start = MagicMock()

    transit_keys = {"username": "some_key", "password": "some_key"}

    tracker_store_validator_instance = mock_tracker_store_validator.return_value
    tracker_store_validator_instance.get_transit_keys_per_endpoint_property = (
        MagicMock()
    )
    tracker_store_validator_instance.get_transit_keys_per_endpoint_property.return_value = (  # noqa: E501
        transit_keys
    )

    endpoint_trait = EndpointTrait(
        endpoint_type="some_endpoint", endpoint_config=EndpointConfig()
    )
    mock_get_vault_endpoint_reader.return_value = tracker_store_validator_instance

    client = MagicMock()
    secrets = dict(
        {
            "username": "some_username",
            "password": "some_password",
        }
    )
    client.secrets.kv.read_secret_version.return_value = dict(
        {"data": dict({"data": secrets})}
    )
    mock_hvac_client.return_value = client

    vault = VaultSecretsManager(host="", token="", secrets_path="")

    mock_decode_transit_secrets = MagicMock()
    monkeypatch.setattr(vault, "_decode_transit_secrets", mock_decode_transit_secrets)

    result = vault.load_secrets(endpoint_trait=endpoint_trait)
    assert result == secrets

    mock_decode_transit_secrets.assert_called_once_with(
        secrets=secrets, transit_keys=transit_keys
    )
    mock_get_vault_endpoint_reader.assert_called_once_with(
        endpoint_trait=endpoint_trait
    )
    tracker_store_validator_instance.get_transit_keys_per_endpoint_property.assert_called_once()


@pytest.mark.usefixtures("mock_hvac_client")
def test_vault_secret_manager_name(
    mock_hvac_client: MagicMock,
    mock_token_manager: MagicMock,
) -> None:
    token_manager_instance = mock_token_manager.return_value
    token_manager_instance.start = MagicMock()
    assert VaultSecretsManager(host="", token="", secrets_path="").name() == "vault"


@pytest.mark.parametrize(
    "host, token, secrets_path, transit_keys, transit_mount_point",
    [
        (
            "some_host",
            "some_token",
            "some_secrets_path",
            {},
            "some_transit_mount_point",
        ),
        ("some_host", "some_token", "some_secrets_path", None, None),
        (
            "some_host",
            "some_token",
            "some_secrets_path",
            dict(
                {
                    "password": "some_transit_key_name",
                    "username": "some_transit_key_name",
                }
            ),
            "some_transit_mount_point",
        ),
    ],
)
def test_vault_secret_manager_init(
    mock_hvac_client: MagicMock,
    mock_token_manager: MagicMock,
    host: Text,
    token: Text,
    secrets_path: Text,
    transit_keys: Optional[Dict[Text, Text]],
    transit_mount_point: Text,
) -> None:
    token_manager_instance = mock_token_manager.return_value
    token_manager_instance.start = MagicMock()

    client = MagicMock()
    secrets = dict(
        {
            "username": "some_username",
            "password": "some_password",
        }
    )
    client.secrets.kv.read_secret_version.return_value = dict(
        {"data": dict({"data": secrets})}
    )
    mock_hvac_client.return_value = client

    vault = VaultSecretsManager(
        host=host,
        token=token,
        secrets_path=secrets_path,
        transit_mount_point=transit_mount_point,
    )
    assert vault.host == host
    assert vault.token == token
    assert vault.secrets_path == secrets_path
    assert vault.transit_mount_point == transit_mount_point
    assert vault.client == client
    assert vault.vault_token_manager == token_manager_instance
    token_manager_instance.start.assert_called_once()


def test_vault_endpoint_reader_get_credentials_location_object() -> None:
    endpoint_config = EndpointConfig(
        username={
            "source": "secrets_manager.vault",
            "secret_key": "some_key",
        }
    )

    processor = VaultEndpointConfigReader(endpoint_config=endpoint_config)
    result = processor.get_credentials_location("username")
    assert isinstance(result, VaultCredentialsLocation)


def test_vault_endpoint_reader_get_credentials_location_mismatched_source(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rasa.core.secrets_manager.endpoints.SUPPORTED_SECRET_MANAGERS", ["aws"]
    )

    endpoint_config = EndpointConfig(
        username={
            "source": "secrets_manager.aws",
            "secret_key": "some_key",
        }
    )

    processor = VaultEndpointConfigReader(endpoint_config=endpoint_config)

    with pytest.raises(RasaException):
        processor.get_credentials_location("username")


@pytest.mark.parametrize(
    "endpoint_config, expected_result",
    [
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": "some_transit_key",
                },
                password={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_password_key",
                    "transit_key": "some_transit_key",
                },
            ),
            {
                "some_username_key": "some_transit_key",
                "some_password_key": "some_transit_key",
            },
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                },
                password={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_password_key",
                },
            ),
            None,
        ),
    ],
)
def test_vault_endpoint_reader_get_transit_keys_per_config_parameter(
    endpoint_config: EndpointConfig, expected_result: Optional[Dict[Text, Text]]
) -> None:
    processor = VaultEndpointConfigReader(endpoint_config=endpoint_config)
    result = processor.get_transit_keys_per_endpoint_property()

    assert result == expected_result


@pytest.mark.parametrize(
    "config_key, expected_result",
    [
        (
            VaultCredentialsLocation(
                source="secrets_manager.vault",
                # deepcode ignore HardcodedNonCryptoSecret/test: Test secret
                secret_key="some_key",
            ),
            True,
        ),
        ("some_key", False),
    ],
)
def test_vault_credentials_location_is_vault_credential_location_instance(
    config_key: Any, expected_result: bool
) -> None:
    assert (
        VaultCredentialsLocation.is_vault_credential_location_instance(config_key)
        == expected_result
    )


@pytest.mark.parametrize(
    "init_values, expected_source, expected_secret_key, expected_transit_key",
    [
        (
            {
                "source": "secrets_manager.vault",
                "secret_key": "some_key",
                "transit_key": "some_transit_key",
            },
            "secrets_manager.vault",
            "some_key",
            "some_transit_key",
        ),
        (
            {
                "source": "secrets_manager.vault",
                "secret_key": "some_key",
                "transit_key": None,
            },
            "secrets_manager.vault",
            "some_key",
            None,
        ),
    ],
)
def test_vault_credentials_location_initialization(
    init_values: Dict[Text, Any],
    expected_source: Text,
    expected_secret_key: Text,
    expected_transit_key: Text,
) -> None:
    secret = VaultCredentialsLocation(**init_values)

    assert secret.source == expected_source
    assert secret.secret_key == expected_secret_key
    assert secret.transit_key == expected_transit_key
