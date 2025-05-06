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
import rasa.studio.download.download
from rasa.studio.config import StudioConfig
from rasa.studio.constants import STUDIO_DOMAIN_FILENAME
from tests.studio.conftest import (
    CALM_CUSTOMIZED_PATTERNS_YAML,
    CALM_ENDPOINTS_YAML,
    encode_yaml,
    get_calm_config_yaml,
    get_calm_domain_yaml,
    get_flows_yaml,
    mock_questionary_text,
)


@pytest.mark.parametrize(
    "overwrite, flow_yaml, domain_file",
    [
        (
            True,
            get_flows_yaml("data/upload/calm/data/flows.yml"),
            STUDIO_DOMAIN_FILENAME,
        ),
        (False, get_flows_yaml("data/upload/calm/data/flows.yml"), "domain.yml"),
        (True, CALM_CUSTOMIZED_PATTERNS_YAML, STUDIO_DOMAIN_FILENAME),
        (False, CALM_CUSTOMIZED_PATTERNS_YAML, "domain.yml"),
    ],
)
def test_handle_download(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    overwrite: bool,
    flow_yaml: str,
    domain_file: str,
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
        rasa.studio.config.StudioConfig,
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
        rasa.studio.download.download.questionary, "confirm", mock_questionary_text
    )

    calm_domain_yaml = get_calm_domain_yaml("data/upload/calm/domain/")
    calm_config_yaml = get_calm_config_yaml("data/upload/calm/config.yml")

    data = {
        "data": {
            "exportAsEncodedYaml": {
                "domain": encode_yaml(calm_domain_yaml),
                "flows": encode_yaml(flow_yaml),
                "config": encode_yaml(calm_config_yaml),
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

    rasa.studio.download.download.handle_download(args)

    studio_domain = get_calm_domain_yaml(tmp_path / domain_file)
    assert studio_domain == calm_domain_yaml
    assert data_path.read_text() == flow_yaml
    assert config_path.read_text() == calm_config_yaml
    assert endpoints_path.read_text() == CALM_ENDPOINTS_YAML
