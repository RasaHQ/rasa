import argparse
import base64
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Set, Text, Union
from unittest.mock import MagicMock

import pytest
import questionary
from pytest import MonkeyPatch

from rasa.shared.exceptions import RasaException
import rasa.studio.upload
from rasa.studio.config import StudioConfig
from rasa.studio.results_logger import with_studio_error_handler, StudioResult
from rasa.studio.upload import make_request
from tests.studio.conftest import (
    CALM_CONFIG_YAML,
    CALM_CUSTOMIZED_PATTERNS_YAML,
    CALM_DOMAIN_YAML,
    CALM_ENDPOINTS_YAML,
    CALM_FLOWS_YAML,
    CALM_NLU_YAML,
    encode_yaml,
    mock_questionary_text,
)


@pytest.mark.parametrize(
    "args, endpoint, expected",
    [
        (
            argparse.Namespace(
                domain="data/upload",
                data="data/upload/data",
                entities=["name"],
                intents=["greet", "inform"],
                config="data/upload/config.yml",
            ),
            "http://studio.amazonaws.com/api/graphql",
            {
                "query": (
                    "mutation ImportFromEncodedYaml"
                    "($input: ImportFromEncodedYamlInput!)"
                    "{\n  importFromEncodedYaml(input: $input)\n}"
                ),
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": (
                            "dmVyc2lvbjogJzMuMScKaW50ZW50czoKLSBncmVldAotIGluZm9ybQplbn"
                            "RpdGllczoKLSBuYW1lOgogICAgcm9sZXM6CiAgICAtIGZpcnN0X25hbWUK"
                            "ICAgIC0gbGFzdF9uYW1lCi0gYWdlCg=="
                        ),
                        "nlu": (
                            "dmVyc2lvbjogIjMuMSIKbmx1OgotIGludGVudDogZ3JlZXQKICBleGFtcGxlc"
                            "zogfAogICAgLSBoZXkKICAgIC0gaGVsbG8KICAgIC0gaGkKICAgIC0gaGVsbG8"
                            "gdGhlcmUKICAgIC0gZ29vZCBtb3JuaW5nCiAgICAtIGdvb2QgZXZlbmluZwogI"
                            "CAgLSBtb2luCiAgICAtIGhleSB0aGVyZQogICAgLSBsZXQncyBnbwogICAgLSB"
                            "oZXkgZHVkZQogICAgLSBnb29kbW9ybmluZwogICAgLSBnb29kZXZlbmluZwogI"
                            "CAgLSBnb29kIGFmdGVybm9vbgotIGludGVudDogaW5mb3JtCiAgZXhhbXBsZXM"
                            "6IHwKICAgIC0gbXkgbmFtZSBpcyBbVXJvc117ImVudGl0eSI6ICJuYW1lIiwgI"
                            "nJvbGUiOiAiZmlyc3RfbmFtZSJ9CiAgICAtIEknbSBbSm9obl17ImVudGl0eSI"
                            "6ICJuYW1lIiwgInJvbGUiOiAiZmlyc3RfbmFtZSJ9CiAgICAtIEhpLCBteSBma"
                            "XJzdCBuYW1lIGlzIFtMdWlzXXsiZW50aXR5IjogIm5hbWUiLCAicm9sZSI6ICJ"
                            "maXJzdF9uYW1lIn0KICAgIC0gTWlsaWNhCiAgICAtIEthcmluCiAgICAtIFN0Z"
                            "XZlbgogICAgLSBJJ20gWzE4XShhZ2UpCiAgICAtIEkgYW0gWzMyXShhZ2UpIHl"
                            "lYXJzIG9sZAogICAgLSA5Cg=="
                        ),
                    }
                },
            },
        ),
        (
            argparse.Namespace(
                assistant_name=["test"],
                calm=True,
                domain="data/upload/calm/domain/domain.yml",
                data=["data/upload/calm/"],
                config="data/upload/calm/config.yml",
                flows="data/upload/flows.yml",
                endpoints="data/upload/calm/endpoints.yml",
            ),
            "http://studio.amazonaws.com/api/graphql",
            {
                "query": (
                    "mutation UploadModernAssistant"
                    "($input: UploadModernAssistantInput!)"
                    "{\n  uploadModernAssistant(input: $input)\n}"
                ),
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": encode_yaml(CALM_DOMAIN_YAML),
                        "flows": encode_yaml(CALM_FLOWS_YAML),
                        "nlu": encode_yaml(CALM_NLU_YAML),
                        "config": (
                            "cmVjaXBlOiBkZWZhdWx0LnYxCmxhbmd1YWdlOiBlbgp"
                            "waXBlbGluZToKLSBuYW1lOiBTaW5nbGVTdGVwTExNQ2"
                            "9tbWFuZEdlbmVyYXRvcgogIGxsbToKICAgIG1vZGVsX"
                            "25hbWU6IGdwdC00CnBvbGljaWVzOgotIG5hbWU6IHJh"
                            "c2EuY29yZS5wb2xpY2llcy5mbG93X3BvbGljeS5GbG9"
                            "3UG9saWN5CmFzc2lzdGFudElkOiBhNWI1ZDNjNS04OG"
                            "NmLTRmZTUtODM1Mi1jNDJlN2NmYWE3YjYK"
                        ),
                        "endpoints": "bmxnOgogIHR5cGU6IHJlcGhyYXNlCg==",
                    }
                },
            },
        ),
        # tests that customized patterns are uploaded only when they are present
        (
            argparse.Namespace(
                assistant_name=["test"],
                calm=True,
                domain="data/upload/calm/domain/domain.yml",
                data=["data/upload/customized_default_flows.yml"],
                config="data/upload/calm/config.yml",
                flows="data/upload/flows.yml",
                endpoints="data/upload/calm/endpoints.yml",
            ),
            "http://studio.amazonaws.com/api/graphql",
            {
                "query": (
                    "mutation UploadModernAssistant"
                    "($input: UploadModernAssistantInput!)"
                    "{\n  uploadModernAssistant(input: $input)\n}"
                ),
                "variables": {
                    "input": {
                        "assistantName": "test",
                        "domain": encode_yaml(CALM_DOMAIN_YAML),
                        "flows": encode_yaml(CALM_CUSTOMIZED_PATTERNS_YAML),
                        "nlu": encode_yaml(""),
                        "config": (
                            "cmVjaXBlOiBkZWZhdWx0LnYxCmxhbmd1YWdlOiBlbgp"
                            "waXBlbGluZToKLSBuYW1lOiBTaW5nbGVTdGVwTExNQ2"
                            "9tbWFuZEdlbmVyYXRvcgogIGxsbToKICAgIG1vZGVsX"
                            "25hbWU6IGdwdC00CnBvbGljaWVzOgotIG5hbWU6IHJh"
                            "c2EuY29yZS5wb2xpY2llcy5mbG93X3BvbGljeS5GbG9"
                            "3UG9saWN5CmFzc2lzdGFudElkOiBhNWI1ZDNjNS04OG"
                            "NmLTRmZTUtODM1Mi1jNDJlN2NmYWE3YjYK"
                        ),
                        "endpoints": "bmxnOgogIHR5cGU6IHJlcGhyYXNlCg==",
                    }
                },
            },
        ),
    ],
)
def test_handle_upload(
    monkeypatch: MonkeyPatch,
    args: argparse.Namespace,
    endpoint: str,
    expected: Dict[str, Any],
) -> None:
    mock = MagicMock()
    mock_token = MagicMock()
    mock_config = MagicMock()
    mock_config.read_config.return_value = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url=endpoint,
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr(rasa.studio.upload, "requests", mock)
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", mock_token)
    monkeypatch.setattr(
        rasa.studio.upload,
        "StudioConfig",
        mock_config,
    )

    monkeypatch.setattr(questionary, "text", mock_questionary_text)

    rasa.studio.upload.handle_upload(args)

    assert mock.post.called
    assert mock.post.call_args[0][0] == endpoint
    assert mock.post.call_args[1]["json"] == expected


@pytest.mark.parametrize(
    "is_calm_bot, mock_fn_name",
    [
        (True, "upload_calm_assistant"),
        (False, "upload_nlu_assistant"),
    ],
)
def test_handle_upload_no_domain_path_specified(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    is_calm_bot: bool,
    mock_fn_name: str,
) -> None:
    """Test the handle_upload function when no domain path is specified in the CLI."""
    # setup test
    assistant_name = "test"
    endpoint = "http://studio.amazonaws.com/api/graphql"
    args = argparse.Namespace(
        assistant_name=[assistant_name],
        # this is the default value when running the cmd without specifying -d flag
        domain="domain.yml",
        config="config.yml",
        calm=is_calm_bot,
    )

    domain_dir = tmp_path / "domain"
    domain_dir.mkdir(parents=True, exist_ok=True)
    domain_path = domain_dir / "domain.yml"
    domain_path.write_text("test domain")

    # default config path
    config_path = tmp_path / "config.yml"
    config_path.write_text("test config")

    domain_paths = [str(domain_dir), str(tmp_path / "domain.yml")]
    # we need to monkeypatch the DEFAULT_DOMAIN_PATHS to be able to use temporary paths
    monkeypatch.setattr(rasa.studio.upload, "DEFAULT_DOMAIN_PATHS", domain_paths)
    monkeypatch.setattr(rasa.studio.upload, "DEFAULT_CONFIG_PATH", str(config_path))

    mock_config = MagicMock()
    mock_config.read_config.return_value = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url=endpoint,
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr(
        rasa.studio.upload,
        "StudioConfig",
        mock_config,
    )

    mock = MagicMock()
    monkeypatch.setattr(rasa.studio.upload, mock_fn_name, mock)

    rasa.studio.upload.handle_upload(args)

    expected_args = argparse.Namespace(
        assistant_name=[assistant_name],
        domain=str(domain_dir),
        config=str(config_path),
        calm=is_calm_bot,
    )

    mock.assert_called_once_with(expected_args, endpoint)


@pytest.mark.parametrize(
    "assistant_name, nlu_examples_yaml, domain_yaml",
    [
        (
            "test",
            dedent(
                """\
                version: '3.1'
                intents:
                - greet
                - inform
                entities:
                - name:
                    roles:
                    - first_name
                    - last_name
                - age"""
            ),
            dedent(
                """\
                version: "3.1"
                nlu:
                - intent: greet
                examples: |
                    - hey
                    - hello
                    - hi
                    - hello there
                    - good morning
                    - good evening
                    - hey there
                    - let's go
                    - hey dude
                    - good afternoon
                - intent: inform
                examples: |
                    - I'm [John]{"entity": "name", "role": "first_name"}
                    - My first name is [Luis]{"entity": "name", "role": "first_name"}
                    - Karin
                    - Steven
                    - I'm [18](age)
                    - I am [32](age) years old"""
            ),
        )
    ],
)
def test_build_request(
    assistant_name: str, nlu_examples_yaml: str, domain_yaml: str
) -> None:
    domain_base64 = base64.b64encode(domain_yaml.encode("utf-8")).decode("utf-8")

    nlu_examples_base64 = base64.b64encode(nlu_examples_yaml.encode("utf-8")).decode(
        "utf-8"
    )

    graphQL_req = rasa.studio.upload.build_request(
        assistant_name, nlu_examples_yaml, domain_yaml
    )

    assert graphQL_req["variables"]["input"]["domain"] == domain_base64
    assert graphQL_req["variables"]["input"]["nlu"] == nlu_examples_base64
    assert graphQL_req["variables"]["input"]["assistantName"] == assistant_name


@pytest.mark.parametrize("assistant_name", ["test"])
def test_build_import_request(assistant_name: str) -> None:
    """Test the build_import_request function.

    :param assistant_name: The name of the assistant
    :return: None
    """
    base64_flows = encode_yaml(CALM_FLOWS_YAML)
    base64_domain = encode_yaml(CALM_DOMAIN_YAML)
    base64_config = encode_yaml(CALM_CONFIG_YAML)
    base64_endpoints = encode_yaml(CALM_ENDPOINTS_YAML)
    base64_nlu = encode_yaml(CALM_NLU_YAML)

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name=assistant_name,
        flows_yaml=CALM_FLOWS_YAML,
        domain_yaml=CALM_DOMAIN_YAML,
        config_yaml=CALM_CONFIG_YAML,
        endpoints=CALM_ENDPOINTS_YAML,
        nlu_yaml=CALM_NLU_YAML,
    )

    assert graphql_req["variables"]["input"]["domain"] == base64_domain
    assert graphql_req["variables"]["input"]["flows"] == base64_flows
    assert graphql_req["variables"]["input"]["assistantName"] == assistant_name
    assert graphql_req["variables"]["input"]["config"] == base64_config
    assert graphql_req["variables"]["input"]["endpoints"] == base64_endpoints
    assert graphql_req["variables"]["input"]["nlu"] == base64_nlu


def test_build_import_request_no_nlu() -> None:
    """Test the build_import_request function when there is no NLU content to upload.

    :return: None
    """
    assistant_name = "test"
    empty_string = ""

    base64_flows = encode_yaml(CALM_FLOWS_YAML)
    base64_domain = encode_yaml(CALM_DOMAIN_YAML)
    base64_config = encode_yaml(empty_string)
    base64_endpoints = encode_yaml(empty_string)

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name,
        flows_yaml=CALM_FLOWS_YAML,
        domain_yaml=CALM_DOMAIN_YAML,
        config_yaml=empty_string,
        endpoints=empty_string,
    )

    assert graphql_req["variables"]["input"]["domain"] == base64_domain
    assert graphql_req["variables"]["input"]["flows"] == base64_flows
    assert graphql_req["variables"]["input"]["assistantName"] == assistant_name
    assert graphql_req["variables"]["input"]["config"] == base64_config
    assert graphql_req["variables"]["input"]["endpoints"] == base64_endpoints
    assert graphql_req["variables"]["input"]["nlu"] == empty_string


@pytest.fixture
def mock_requests(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("rasa.studio.upload.requests", mock)
    return mock


@pytest.fixture
def mock_keycloak_token(monkeypatch):
    mock = MagicMock()
    mock.get_token.return_value.token_type = "Bearer"
    mock.get_token.return_value.access_token = "mock_token"
    monkeypatch.setattr("rasa.studio.upload.KeycloakTokenReader", lambda: mock)
    return mock


@pytest.mark.parametrize(
    "query_type, response_data, status_code, expected_result",
    [
        (
            "ImportFromEncodedYaml",
            {"data": {"importFromEncodedYaml": ""}},
            200,
            StudioResult("Upload successful", True),
        ),
        (
            "ImportFromEncodedYaml",
            {"errors": [{"message": "Upload failed with status code 405"}]},
            405,
            StudioResult(
                "Upload failed with status code 405",
                False,
            ),
        ),
        (
            "UploadModernAssistant",
            {"data": {"uploadModernAssistant": ""}},
            200,
            StudioResult("Upload successful", True),
        ),
        (
            "UploadModernAssistant",
            {"errors": [{"message": "Error 1"}, {"message": "Error 2"}]},
            500,
            StudioResult("Error 1; Error 2", False),
        ),
    ],
)
def test_make_request(
    mock_requests,
    mock_keycloak_token,
    query_type,
    response_data,
    status_code,
    expected_result,
):
    # Arrange
    endpoint = "http://studio.test/api/graphql/"
    graphql_req = {
        "query": f"mutation {query_type}"
        f"($input: {query_type}Input!) "
        f"{{\n  {query_type.lower()}"
        f"(input: $input)\n}}",
        "variables": {
            "input": {
                "assistantName": "test",
                "domain": "base64_encoded_domain",
                "nlu": "base64_encoded_nlu",
            }
        },
    }

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_data
    mock_requests.post.return_value = mock_response

    # Act
    @with_studio_error_handler
    def test_make_request_func():
        return make_request(endpoint, graphql_req)

    result = test_make_request_func()

    # Assert
    assert isinstance(result, StudioResult)
    assert result.message == expected_result.message
    assert result.was_successful == expected_result.was_successful

    mock_requests.post.assert_called_once_with(
        endpoint,
        json=graphql_req,
        headers={
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json",
        },
    )
    mock_keycloak_token.get_token.assert_called_once()


@pytest.mark.parametrize(
    "domain_from_files, intents, entities, expected_domain",
    [
        (
            {
                "version": "3.1",
                "intents": [
                    "greet",
                    "inform",
                    "goodbye",
                    "deny",
                ],
                "entities": [
                    {"name": {"roles": ["first_name", "last_name"]}},
                    "age",
                    "destination",
                    "origin",
                ],
            },
            ["greet", "inform"],
            ["name"],
            {
                "version": "3.1",
                "intents": [
                    "greet",
                    "inform",
                ],
                "entities": [{"name": {"roles": ["first_name", "last_name"]}}],
            },
        ),
    ],
)
def test_filter_domain(
    domain_from_files: Dict[str, Any],
    intents: List[str],
    entities: List[Union[str, Dict[Any, Any]]],
    expected_domain: Dict[str, Any],
) -> None:
    filtered_domain = rasa.studio.upload._filter_domain(
        domain_from_files=domain_from_files, intents=intents, entities=entities
    )
    assert filtered_domain == expected_domain


@pytest.mark.parametrize(
    "intents, entities, found_intents, found_entities",
    [
        (
            ["greet", "inform"],
            ["name"],
            ["greet", "goodbye", "deny"],
            ["name", "destination", "origin"],
        ),
    ],
)
def test_check_for_missing_primitives(
    intents: List[str],
    entities: List[str],
    found_intents: List[str],
    found_entities: List[str],
) -> None:
    with pytest.raises(RasaException) as excinfo:
        rasa.studio.upload._check_for_missing_primitives(
            intents, entities, found_intents, found_entities
        )
        assert "The following intents were not found in the domain: inform" in str(
            excinfo.value
        )
        assert "The following entities were not found in the domain: age" in str(
            excinfo.value
        )


@pytest.mark.parametrize(
    "args, intents_from_files, entities_from_files, "
    "expected_intents, expected_entities",
    [
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents=None,
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["goodbye", "greet", "deny"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents={},
                entities={"name"},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["goodbye", "greet", "deny"],
            ["name"],
        ),
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities=None,
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["destination", "name", "origin"],
        ),
        (
            argparse.Namespace(
                intents={"greet", "inform"},
                entities={},
            ),
            {"greet", "goodbye", "deny"},
            {"name", "destination", "origin"},
            ["greet", "inform"],
            ["destination", "name", "origin"],
        ),
    ],
)
def test_get_selected_entities_and_intents(
    args: argparse.Namespace,
    intents_from_files: Set[Text],
    entities_from_files: List[Text],
    expected_intents: List[Text],
    expected_entities: List[Text],
) -> None:
    entities, intents = rasa.studio.upload._get_selected_entities_and_intents(
        args=args,
        intents_from_files=intents_from_files,
        entities_from_files=entities_from_files,
    )

    assert intents.sort() == expected_intents.sort()
    assert entities.sort() == expected_entities.sort()
