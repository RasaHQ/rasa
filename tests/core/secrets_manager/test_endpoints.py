# file deepcode ignore HardcodedNonCryptoSecret/test: Secrets are all just examples for tests. # noqa: E501

from typing import Any, Dict, List, Text, Union

import pytest
from rasa.utils.endpoints import EndpointConfig

from rasa.core.secrets_manager.endpoints import (
    CredentialsLocation,
    EndpointReader,
    TrackerStoreEndpointValidator,
)


@pytest.mark.parametrize(
    "endpoint_config, expected_result",
    [
        (EndpointConfig(password="some value"), True),
        (
            EndpointConfig(
                username="some value",
                password="some value",
            ),
            True,
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                },
                password="aaa",
            ),
            True,
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": "some_transit_key",
                },
                password="aaa",
            ),
            True,
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": None,
                },
                password={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_password_key",
                    "transit_key": "some transit key",
                },
            ),
            True,
        ),
        (EndpointConfig(username="some value"), False),
        (EndpointConfig(username=""), False),
        (EndpointConfig(password=""), False),
        (
            EndpointConfig(
                username="",
                password="",
            ),
            False,
        ),
        (
            EndpointConfig(
                username={
                    "source": "",
                    "secret_key": "some_username_key",
                }
            ),
            False,
        ),
        (
            EndpointConfig(
                password={
                    "secret_key": "some_username_key",
                }
            ),
            False,
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "",
                }
            ),
            False,
        ),
        (
            EndpointConfig(
                password={
                    "source": "",
                    "secret_key": "some_username_key",
                }
            ),
            False,
        ),
        (
            EndpointConfig(
                password={
                    "source": "secrets_manager.vault",
                    "secret_key": "",
                }
            ),
            False,
        ),
        (
            EndpointConfig(),
            False,
        ),
    ],
)
def test_tracker_store_endpoint_validator_is_endpoint_config_valid(
    endpoint_config: EndpointConfig,
    expected_result: bool,
) -> None:
    assert (
        TrackerStoreEndpointValidator(
            EndpointReader(endpoint_config)
        ).is_endpoint_config_valid()
        == expected_result
    )


@pytest.mark.parametrize(
    "raw_credentials, expected_result",
    [
        (
            {
                "source": "secrets_manager.vault",
            },
            True,
        ),
        (
            {
                "source": ".vault",
            },
            False,
        ),
        (
            {
                "source": "secrets_manager.",
            },
            False,
        ),
        (
            {
                "source": "",
            },
            False,
        ),
        (
            {
                "source": "aaaaa",
            },
            False,
        ),
    ],
)
def test_credentials_location_is_source_valid(
    raw_credentials: Dict[Text, Any],
    expected_result: bool,
) -> None:
    assert (
        CredentialsLocation._is_source_valid(raw_credentials_location=raw_credentials)
        == expected_result
    )


@pytest.mark.parametrize(
    "raw_credentials_location, expected_result",
    [
        (
            {
                "username": "some value",
                # deepcode ignore NoHardcodedPasswords/test: Test credential
                "password": "some value",
            },
            False,
        ),
        (
            {
                "source": "secrets_manager.vault",
                "secret_key": "some_username_key",
            },
            True,
        ),
        (
            {
                "source": "secrets_manager.vault",
                "secret_key": "some_username_key",
                "transit_key": "some_transit_key",
            },
            True,
        ),
        (
            {
                "source": "secrets_manager.vault",
                "secret_key": "some_username_key",
                "transit_key": None,
            },
            True,
        ),
    ],
)
def test_credentials_location_is_credentials_location_valid(
    raw_credentials_location: Dict[Text, Text], expected_result: bool
) -> None:
    assert (
        CredentialsLocation.is_credentials_location_valid(raw_credentials_location)
        == expected_result
    )


@pytest.mark.parametrize(
    "raw_credentials_location, property_name, expected_result",
    [
        (
            {
                "source": None,
                "secret_key": "some_username_key",
            },
            "source",
            False,
        ),
        (
            {
                "source": "",
                "secret_key": "some_username_key",
            },
            "source",
            False,
        ),
        (
            {
                "source": "secrets_manager.vault",
                "secret_key": "some_username_key",
            },
            "source",
            True,
        ),
        (
            {
                "source": "secrets_manager.vault",
                "secret_key": "some_username_key",
                "transit_key": "some_transit_key",
            },
            "secret_key",
            True,
        ),
        (
            {
                "source": "secrets_manager.vault",
                "secret_key": "some_username_key",
                "transit_key": None,
            },
            "secret_key",
            True,
        ),
    ],
)
def test_credentials_location_is_property_valid(
    raw_credentials_location: Dict[Text, Text],
    property_name: Text,
    expected_result: bool,
) -> None:
    assert (
        CredentialsLocation.is_property_non_empty_string(
            raw_credentials_location, property_name
        )
        == expected_result
    )


@pytest.mark.parametrize(
    "endpoint_config, config_key_name, expected_result",
    [
        (EndpointConfig(username="some value"), "username", "some value"),
        (EndpointConfig(password="some value"), "password", "some value"),
        (
            EndpointConfig(
                username="some value",
                password="some value 2",
            ),
            "username",
            "some value",
        ),
    ],
)
def test_endpoint_reader_get_config_key_string(
    endpoint_config: EndpointConfig,
    config_key_name: Text,
    expected_result: Union[str, CredentialsLocation],
) -> None:
    result = EndpointReader(endpoint_config=endpoint_config).get_property_value(
        endpoint_property_name=config_key_name
    )
    assert result == expected_result


@pytest.mark.parametrize(
    "endpoint_config, config_key_name, expected_result",
    [
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                }
            ),
            "username",
            CredentialsLocation(
                source="secrets_manager.vault", secret_key="some_username_key"
            ),
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": "some_transit_key",
                }
            ),
            "username",
            CredentialsLocation(
                source="secrets_manager.vault",
                secret_key="some_username_key",
                transit_key="some_transit_key",
            ),
        ),
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
            "password",
            CredentialsLocation(
                source="secrets_manager.vault",
                secret_key="some_password_key",
                transit_key="some_transit_key",
            ),
        ),
    ],
)
def test_endpoint_reader_get_config_key_secret_manager_stored_secret(
    endpoint_config: EndpointConfig,
    config_key_name: Text,
    expected_result: CredentialsLocation,
) -> None:
    result = EndpointReader(endpoint_config=endpoint_config).get_property_value(
        endpoint_property_name=config_key_name
    )

    assert isinstance(result, CredentialsLocation)
    assert result.source == expected_result.source
    assert result.secret_key == expected_result.secret_key
    assert result.kwargs == expected_result.kwargs


@pytest.mark.parametrize(
    "endpoint_config, config_key_name, expected_result",
    [
        (
            EndpointConfig(),
            "url",
            True,
        ),
        (
            EndpointConfig(),
            "token",
            True,
        ),
        (
            EndpointConfig(),
            "token_name",
            True,
        ),
        (
            EndpointConfig(),
            "params",
            True,
        ),
        (
            EndpointConfig(),
            "headers",
            True,
        ),
        (
            EndpointConfig(),
            "basic_auth",
            True,
        ),
        (
            EndpointConfig(),
            "cafile",
            True,
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                }
            ),
            "username",
            True,
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                }
            ),
            "password",
            False,
        ),
        (
            EndpointConfig(),
            "random_key",
            False,
        ),
    ],
)
def test_endpoint_reader_config_has_key(
    endpoint_config: EndpointConfig, config_key_name: Text, expected_result: bool
) -> None:
    assert (
        EndpointReader(endpoint_config=endpoint_config).has_property(config_key_name)
        == expected_result
    )


@pytest.mark.parametrize(
    "endpoint_config, config_key",
    [
        (EndpointConfig(username="some value"), "username"),
        (EndpointConfig(password="some value"), "password"),
        (
            EndpointConfig(
                username="some value",
                password="some value",
            ),
            "username",
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                }
            ),
            "username",
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": "some_transit_key",
                }
            ),
            "username",
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": None,
                },
                password={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_password_key",
                    "transit_key": "some transit key",
                },
            ),
            "password",
        ),
    ],
)
def test_endpoint_reader_validate_config_value(
    endpoint_config: EndpointConfig, config_key: Text
) -> None:
    assert EndpointReader(endpoint_config=endpoint_config).is_endpoint_property_valid(
        endpoint_property=config_key
    )


@pytest.mark.parametrize(
    "endpoint_config, config_key",
    [
        (EndpointConfig(username=""), "username"),
        (EndpointConfig(password=""), "password"),
        (
            EndpointConfig(
                username="",
                password="some value",
            ),
            "username",
        ),
        (
            EndpointConfig(
                username={
                    "source": "",
                    "secret_key": "some_username_key",
                }
            ),
            "username",
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "",
                    "transit_key": "some_transit_key",
                }
            ),
            "username",
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": None,
                },
                password={
                    "source": "",
                    "secret_key": "some_password_key",
                    "transit_key": "some transit key",
                },
            ),
            "password",
        ),
    ],
)
def test_endpoint_reader_validate_config_value_invalid_config(
    endpoint_config: EndpointConfig, config_key: Text
) -> None:
    assert (
        EndpointReader(endpoint_config=endpoint_config).is_endpoint_property_valid(
            endpoint_property=config_key
        )
    ) is False


@pytest.mark.parametrize(
    "endpoint_config, expected_result",
    [
        (EndpointConfig(username="some value"), None),
        (EndpointConfig(password="some value"), None),
        (
            EndpointConfig(
                username="some value",
                password="some value",
            ),
            None,
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                }
            ),
            {"vault": ["username"]},
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": "some_transit_key",
                }
            ),
            {"vault": ["username"]},
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_username_key",
                    "transit_key": None,
                },
                password={
                    "source": "secrets_manager.vault",
                    "secret_key": "some_password_key",
                    "transit_key": "some transit key",
                },
            ),
            {"vault": ["username", "password"]},
        ),
    ],
)
def test_endpoint_reader_get_config_keys_managed_by_secret_manager(
    endpoint_config: EndpointConfig, expected_result: Dict
) -> None:
    assert (
        EndpointReader(
            endpoint_config=endpoint_config
        ).get_endpoint_properties_managed_by_secret_manager()
        == expected_result
    )


@pytest.mark.parametrize(
    "config_key, expected_result",
    [
        (
            CredentialsLocation(
                source="secrets_manager.vault",
                secret_key="some_key",
            ),
            True,
        ),
        ("some_key", False),
    ],
)
def test_credentials_location_is_config_key_a_credentials_location(
    config_key: Any, expected_result: bool
) -> None:
    assert (
        CredentialsLocation.is_credentials_location_instance(config_key)
        == expected_result
    )


@pytest.mark.parametrize(
    "endpoint_config, expected_result",
    [
        (
            EndpointConfig(),
            ["url", "token", "token_name", "params", "headers", "cafile", "basic_auth"],
        ),
        (
            EndpointConfig(password="some value"),
            [
                "url",
                "token",
                "token_name",
                "params",
                "headers",
                "cafile",
                "basic_auth",
                "password",
            ],
        ),
        (
            EndpointConfig(username="username", password="some value"),
            [
                "url",
                "token",
                "token_name",
                "params",
                "headers",
                "cafile",
                "basic_auth",
                "password",
                "username",
            ],
        ),
    ],
)
def test_endpoint_reader_get_keys(
    endpoint_config: EndpointConfig, expected_result: List[Text]
) -> None:
    assert (
        EndpointReader(endpoint_config=endpoint_config).get_keys().sort()
    ) == expected_result.sort()
