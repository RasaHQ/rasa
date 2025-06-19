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
from rasa.shared.constants import DEFAULT_DATA_PATH
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader
from rasa.studio.config import StudioConfig
from rasa.studio.constants import DOMAIN_FILENAME
from rasa.studio.pull.data import STUDIO_FLOWS_DIR_NAME
from tests.studio.conftest import (
    CALM_CUSTOMIZED_PATTERNS_YAML,
    CALM_ENDPOINTS_YAML,
    encode_yaml,
    get_calm_config_yaml,
    get_calm_domain_yaml,
    get_flows_yaml,
)


@pytest.mark.parametrize(
    "flow_yaml, domain_file",
    [
        (
            get_flows_yaml("data/upload/calm/data/flows.yml"),
            DOMAIN_FILENAME,
        ),
        (CALM_CUSTOMIZED_PATTERNS_YAML, DOMAIN_FILENAME),
    ],
)
def test_handle_download(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    flow_yaml: str,
    domain_file: str,
) -> None:
    mock_config = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url="http://studio.amazonaws.com/api/graphql",
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr(
        StudioConfig,
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

    assistant_name = "calm"
    args = argparse.Namespace(assistant_name=assistant_name)
    monkeypatch.chdir(tmp_path)
    rasa.studio.download.handle_download(args)

    downloaded_assistant = tmp_path / assistant_name
    assert (downloaded_assistant / DOMAIN_FILENAME).read_text() == calm_domain_yaml
    assert (downloaded_assistant / "config.yml").read_text() == calm_config_yaml
    assert (downloaded_assistant / "endpoints.yml").read_text() == CALM_ENDPOINTS_YAML

    flows_dir = downloaded_assistant / DEFAULT_DATA_PATH / STUDIO_FLOWS_DIR_NAME
    assert flows_dir.exists()

    flows_list = YAMLFlowsReader.read_from_string(flow_yaml)
    for flow in flows_list.underlying_flows:
        flow_file = flows_dir / f"{flow.id}.yml"
        assert flow_file.exists()

        downloaded_flow = YAMLFlowsReader.read_from_file(flow_file).underlying_flows[0]
        assert flow == downloaded_flow
