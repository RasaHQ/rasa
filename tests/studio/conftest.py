from unittest.mock import MagicMock

import pytest

from rasa.studio.auth import StudioAuth
from rasa.studio.config import StudioConfig


@pytest.fixture
def mock_keycloak_instance_object() -> MagicMock:
    keycloak_open_id_instance = MagicMock()
    keycloak_open_id_instance.server_url = "http://localhost:8080"
    keycloak_open_id_instance.client_id = "client_id"
    keycloak_open_id_instance.realm_name = "realm_name"
    keycloak_open_id_instance.token = MagicMock()
    return keycloak_open_id_instance


@pytest.fixture
def mock_keycloak_instance(
    mock_keycloak_open_id: MagicMock, mock_keycloak_instance_object: MagicMock
) -> MagicMock:
    mock_keycloak_open_id.return_value = mock_keycloak_instance_object
    return mock_keycloak_instance_object


@pytest.fixture
def studio_auth(mock_keycloak_instance: MagicMock) -> StudioAuth:
    config = StudioConfig(
        authentication_server_url="http://localhost:8080",
        studio_url="http://localhost:8080/graphql",
        client_id="client_id",
        realm_name="realm_name",
    )
    auth = StudioAuth(config)
    return auth
