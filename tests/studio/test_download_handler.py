import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from pytest import MonkeyPatch
from requests import Response

import rasa.studio.auth
import rasa.studio.data_handler
import rasa.studio.download
from rasa.studio.config import StudioConfig
from tests.studio.conftest import (
    CALM_CONFIG_YAML,
    CALM_CUSTOMIZED_PATTERNS_YAML,
    CALM_DOMAIN_YAML,
    CALM_ENDPOINTS_YAML,
    CALM_FLOWS_YAML,
    encode_yaml,
    mock_questionary_text,
)


@pytest.mark.parametrize(
    "overwrite, flow_yaml",
    [
        (True, CALM_FLOWS_YAML),
        (False, CALM_FLOWS_YAML),
        (True, CALM_CUSTOMIZED_PATTERNS_YAML),
        (False, CALM_CUSTOMIZED_PATTERNS_YAML),
    ],
)
def test_handle_download(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    overwrite: bool,
    flow_yaml: str,
) -> None:
    domain_path = tmp_path / "domain.yml"
    domain_path.touch()

    data_path = tmp_path / "flows.yml"
    data_path.touch()
    assert data_path.read_text() == ""

    config_path = tmp_path / "config.yml"
    config_path.touch()

    endpoints_path = tmp_path / "endpoints.yml"
    endpoints_path.touch()

    args = argparse.Namespace(
        domain=str(domain_path),
        data=[str(data_path)],
        config=str(config_path),
        endpoints=str(endpoints_path),
        assistant_name="calm",
        overwrite=overwrite,
    )
    mock_config = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url="http://studio.amazonaws.com/api/graphql",
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr(
        rasa.studio.download.StudioConfig,
        "read_config",
        lambda *args: mock_config,
    )

    mock_token = Mock(return_value="token")
    monkeypatch.setattr(
        rasa.studio.auth.KeycloakTokenReader, "__init__", lambda *args: None
    )
    monkeypatch.setattr(
        rasa.studio.auth.KeycloakTokenReader, "get_token", lambda *args: mock_token
    )

    monkeypatch.setattr(
        rasa.studio.download.questionary, "confirm", mock_questionary_text
    )

    data = {
        "data": {
            "exportAsEncodedYaml": {
                "domain": encode_yaml(CALM_DOMAIN_YAML),
                "flows": encode_yaml(flow_yaml),
                "config": encode_yaml(CALM_CONFIG_YAML),
                "endpoints": encode_yaml(CALM_ENDPOINTS_YAML),
            }
        },
    }

    stub_response = Response()
    stub_response.status_code = 200
    stub_response._content = json.dumps(data).encode("utf-8")

    monkeypatch.setattr(
        rasa.studio.data_handler.requests, "post", MagicMock(return_value=stub_response)
    )

    rasa.studio.download.handle_download(args)

    assert data_path.read_text() == flow_yaml
    assert domain_path.read_text() == CALM_DOMAIN_YAML
    assert config_path.read_text() == CALM_CONFIG_YAML
    assert endpoints_path.read_text() == CALM_ENDPOINTS_YAML
