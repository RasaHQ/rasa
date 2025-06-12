import argparse
import base64
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Set, Text, Union
from unittest.mock import MagicMock, patch

import pytest
import questionary
from pytest import MonkeyPatch

import rasa.shared.utils.io
import rasa.shared.utils.yaml
import rasa.studio.upload
from rasa.shared.exceptions import RasaException
from rasa.studio.config import StudioConfig
from rasa.studio.results_logger import StudioResult, with_studio_error_handler
from rasa.studio.upload import (
    build_get_assistant_by_name_request,
    check_if_assistant_already_exists,
    make_request,
)
from tests.studio.conftest import (
    CALM_ENDPOINTS_YAML,
    CALM_NLU_YAML,
    encode_yaml,
    get_calm_config_yaml,
    get_calm_domain_yaml,
    get_flows_yaml,
    mock_questionary_text,
)


@pytest.mark.parametrize(
    "args, endpoint, expected",
    [
        (
            argparse.Namespace(
                assistant_name="test",
                calm=True,
                domain=Path("data/upload/calm/domain/"),
                data=[Path("data/upload/calm/data/")],
                config=Path("data/upload/calm/config.yml"),
                endpoints=Path("data/upload/calm/endpoints.yml"),
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
                        "domain": encode_yaml(
                            get_calm_domain_yaml(Path("data/upload/calm/domain/"))
                        ),
                        "flows": encode_yaml(
                            get_flows_yaml(Path("data/upload/calm/data/"))
                        ),
                        "nlu": encode_yaml(CALM_NLU_YAML),
                        "config": encode_yaml(
                            get_calm_config_yaml(Path("data/upload/calm/config.yml"))
                        ),
                        "endpoints": "bmxnOgogIHR5cGU6IHJlcGhyYXNlCg==",
                    }
                },
            },
        ),
        # tests that customized patterns are uploaded only when they are present
        (
            argparse.Namespace(
                assistant_name="test",
                calm=True,
                domain=Path("data/upload/calm/domain/domain.yml"),
                data=[
                    Path("data/upload/customized_default_flows.yml"),
                ],
                config=Path("data/upload/calm/config.yml"),
                endpoints=Path("data/upload/calm/endpoints.yml"),
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
                        "domain": encode_yaml(
                            get_calm_domain_yaml(
                                Path("data/upload/calm/domain/domain.yml")
                            )
                        ),
                        "flows": encode_yaml(
                            get_flows_yaml(
                                Path("data/upload/customized_default_flows.yml")
                            )
                        ),
                        "config": encode_yaml(
                            get_calm_config_yaml(Path("data/upload/calm/config.yml"))
                        ),
                        "endpoints": "bmxnOgogIHR5cGU6IHJlcGhyYXNlCg==",
                    }
                },
            },
        ),
        # test when endpoints.yml contain an environment variable
        (
            argparse.Namespace(
                assistant_name="test",
                calm=True,
                domain=Path("data/upload/calm/domain/"),
                data=[Path("data/upload/calm/data/")],
                config=Path("data/upload/calm/config.yml"),
                endpoints=Path("data/upload/endpoints_with_env_var.yml"),
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
                        "domain": encode_yaml(
                            get_calm_domain_yaml(Path("data/upload/calm/domain/"))
                        ),
                        "flows": encode_yaml(get_flows_yaml(Path("data/upload/calm"))),
                        "nlu": encode_yaml(CALM_NLU_YAML),
                        "config": encode_yaml(
                            get_calm_config_yaml(Path("data/upload/calm/config.yml"))
                        ),
                        "endpoints": encode_yaml(
                            rasa.shared.utils.io.read_file(
                                Path("data/upload/endpoints_with_env_var.yml")
                            )
                        ),
                    }
                },
            },
        ),
        # test with domain as directory
        (
            argparse.Namespace(
                assistant_name="test",
                calm=True,
                domain=Path("data/upload/simple_bot_with_domain_directory/domain"),
                data=[Path("data/upload/simple_bot_with_domain_directory/data/")],
                config=Path("data/upload/calm/config.yml"),
                endpoints=Path("data/upload/endpoints_with_env_var.yml"),
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
                        "domain": encode_yaml(
                            get_calm_domain_yaml(
                                Path(
                                    "data/upload/simple_bot_with_domain_directory/domain"
                                )
                            )
                        ),
                        "flows": encode_yaml(
                            get_flows_yaml(
                                Path(
                                    "data/upload/simple_bot_with_domain_directory/data/"
                                )
                            )
                        ),
                        "config": encode_yaml(
                            get_calm_config_yaml(Path("data/upload/calm/config.yml"))
                        ),
                        "endpoints": encode_yaml(
                            rasa.shared.utils.io.read_file(
                                Path("data/upload/endpoints_with_env_var.yml")
                            )
                        ),
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
    mock_replace_environment_variables: MagicMock,
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

    mock_replace_environment_variables.assert_not_called()

    assert mock.post.called
    assert mock.post.call_args[0][0] == endpoint
    assert mock.post.call_args[1]["verify"] is True
    assert mock.post.call_args[1]["json"] == expected


@pytest.mark.parametrize("disable_verify", [True, False])
def test_handle_upload_no_domain_path_specified(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    disable_verify: bool,
) -> None:
    """Test the handle_upload function when no domain path is specified in the CLI."""
    # setup test
    assistant_name = "test"
    endpoint = "http://studio.amazonaws.com/api/graphql"
    args = argparse.Namespace(
        assistant_name=[assistant_name],
        # this is the default value when running the cmd without specifying -d flag
        domain=None,
        config=None,
        calm=True,
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
    monkeypatch.setattr(rasa.studio.upload, "requests", MagicMock())

    mock_config = MagicMock()
    mock_config.read_config.return_value = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url=endpoint,
        realm_name="rasa-test",
        client_id="rasa-cli",
        disable_verify=disable_verify,
    )
    monkeypatch.setattr(
        rasa.studio.upload,
        "StudioConfig",
        mock_config,
    )
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", MagicMock())

    mock = MagicMock()
    monkeypatch.setattr(rasa.studio.upload, "upload_calm_assistant", mock)

    rasa.studio.upload.handle_upload(args)

    expected_args = argparse.Namespace(
        assistant_name=[assistant_name],
        domain=str(domain_dir),
        config=str(config_path),
        calm=True,
    )

    mock.assert_called_once_with(expected_args, endpoint, verify=not disable_verify)


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
    calm_flows_yaml = get_flows_yaml("data/upload/calm/data/flows.yml")
    calm_domain_yaml = get_calm_domain_yaml("data/upload/calm/domain/")
    calm_config_yaml = get_calm_config_yaml("data/upload/calm/config.yml")

    base64_flows = encode_yaml(calm_flows_yaml)
    base64_domain = encode_yaml(calm_domain_yaml)
    base64_config = encode_yaml(calm_config_yaml)
    base64_endpoints = encode_yaml(CALM_ENDPOINTS_YAML)
    base64_nlu = encode_yaml(CALM_NLU_YAML)

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name=assistant_name,
        flows_yaml=calm_flows_yaml,
        domain_yaml=calm_domain_yaml,
        config_yaml=calm_config_yaml,
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

    calm_flows_yaml = get_flows_yaml("data/upload/calm/data/flows.yml")
    calm_domain_yaml = get_calm_domain_yaml("data/upload/calm/domain/")

    base64_flows = encode_yaml(calm_flows_yaml)
    base64_domain = encode_yaml(calm_domain_yaml)
    base64_config = encode_yaml(empty_string)
    base64_endpoints = encode_yaml(empty_string)

    graphql_req = rasa.studio.upload.build_import_request(
        assistant_name,
        flows_yaml=calm_flows_yaml,
        domain_yaml=calm_domain_yaml,
        config_yaml=empty_string,
        endpoints=empty_string,
    )

    assert graphql_req["variables"]["input"]["domain"] == base64_domain
    assert graphql_req["variables"]["input"]["flows"] == base64_flows
    assert graphql_req["variables"]["input"]["assistantName"] == assistant_name
    assert graphql_req["variables"]["input"]["config"] == base64_config
    assert graphql_req["variables"]["input"]["endpoints"] == base64_endpoints
    assert "nlu" not in graphql_req["variables"]["input"]


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
        verify=True,
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


def test_check_if_assistant_already_exists(monkeypatch: MonkeyPatch):
    mock_token = MagicMock()
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", mock_token)

    assistant_name = "test_assistant"
    endpoint = "https://studio.example.com/graphql"
    verify = True

    # Mock response for when the assistant exists
    mock_response_exists = MagicMock()
    mock_response_exists.json.return_value = {
        "data": {
            "assistantByName": {"id": "123", "name": assistant_name, "mode": "test"}
        }
    }

    # Mock response for when the assistant does not exist
    mock_response_not_exists = MagicMock()
    mock_response_not_exists.json.return_value = {"data": {"assistantByName": None}}

    with patch("rasa.studio.upload.requests.post") as mock_post:
        # Assistant exists
        mock_post.return_value = mock_response_exists
        assert (
            check_if_assistant_already_exists(assistant_name, endpoint, verify) is True
        )

        # Assistant does not exist
        mock_post.return_value = mock_response_not_exists
        assert (
            check_if_assistant_already_exists(assistant_name, endpoint, verify) is False
        )


def test_build_get_assistant_by_name_request():
    assistant_name = "test_assistant"
    expected_request = {
        "query": (
            "query AssistantByName($input: AssistantByNameInput!) {"
            " assistantByName(input: $input) {"
            " ... on Assistant { id name mode }"
            " ... on AssistantByName_AssistantNotFound { _ }"
            " }"
            "}"
        ),
        "variables": {
            "input": {
                "assistantName": assistant_name,
            }
        },
    }

    result = build_get_assistant_by_name_request(assistant_name)
    assert result == expected_request


def test_build_import_request_skips_none_values() -> None:
    assistant_name = "test_assistant"
    sample_flows = "flows:"
    sample_domain = "responses:"
    req = rasa.studio.upload.build_import_request(
        assistant_name=assistant_name,
        flows_yaml=sample_flows,
        domain_yaml=sample_domain,
        config_yaml=None,
        endpoints=None,
        nlu_yaml=None,
    )

    payload = req["variables"]["input"]

    # mandatory fields are present and encoded
    assert payload["assistantName"] == assistant_name
    assert payload["flows"] == encode_yaml(sample_flows)
    assert payload["domain"] == encode_yaml(sample_domain)

    # all fields that were passed as None must be absent
    assert "config" not in payload
    assert "endpoints" not in payload
    assert "nlu" not in payload
