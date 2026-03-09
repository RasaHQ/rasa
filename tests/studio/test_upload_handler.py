import argparse
import base64
from pathlib import Path
from sys import modules
from textwrap import dedent
from types import SimpleNamespace
from typing import Any, Dict, List, Set, Text, Union
from unittest.mock import MagicMock, patch

import pytest
import questionary
from pytest import MonkeyPatch

import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.shared.utils.yaml
import rasa.studio.upload
from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.dialogue_understanding.generator import SingleStepLLMCommandGenerator
from rasa.shared.constants import DEFAULT_PROMPTS_PATH
from rasa.shared.exceptions import RasaException
from rasa.studio.config import StudioConfig
from rasa.studio.prompts import (
    COMMAND_GENERATOR_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_NAME,
    ENTERPRISE_SEARCH_NAME,
)
from rasa.studio.results_logger import StudioResult, with_studio_error_handler
from rasa.studio.upload import (
    build_delete_assistant_request,
    build_get_assistant_by_name_request,
    check_if_assistant_already_exists,
    get_assistant_id_by_name,
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
                data=Path("data/upload/calm/data/"),
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
                data=Path("data/upload/customized_default_flows.yml"),
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
                data=Path("data/upload/calm/data/"),
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
                data=Path("data/upload/simple_bot_with_domain_directory/data/"),
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
    mock_post = MagicMock()
    mock_post.return_value = MagicMock()
    mock_token = MagicMock()
    mock_config = MagicMock()
    mock_config.read_config.return_value = StudioConfig(
        authentication_server_url="http://studio.amazonaws.com",
        studio_url=endpoint,
        realm_name="rasa-test",
        client_id="rasa-cli",
    )
    monkeypatch.setattr("rasa.studio.upload.requests.Session.post", mock_post)
    monkeypatch.setattr("rasa.studio.upload.KeycloakTokenReader", mock_token)
    monkeypatch.setattr(
        rasa.studio.upload,
        "StudioConfig",
        mock_config,
    )

    monkeypatch.setattr(questionary, "text", mock_questionary_text)

    rasa.studio.upload.handle_upload(args)

    mock_replace_environment_variables.assert_not_called()

    assert mock_post.called
    assert mock_post.call_args[0][0] == endpoint
    assert mock_post.call_args[1]["verify"] is True

    actual = mock_post.call_args[1]["json"]
    assert actual["query"] == expected["query"]

    actual_input = actual["variables"]["input"]
    expected_input = expected["variables"]["input"]

    # Compare stable fields directly
    for key in ["assistantName", "domain", "flows", "config", "endpoints"]:
        if key in expected_input:
            assert actual_input[key] == expected_input[key]

    # Compare NLU semantically (decoded YAML), tolerant to quote style
    if "nlu" in expected_input:
        actual_nlu_yaml = base64.b64decode(actual_input["nlu"]).decode("utf-8")
        expected_nlu_yaml = base64.b64decode(expected_input["nlu"]).decode("utf-8")

        actual_nlu = rasa.shared.utils.yaml.read_yaml(actual_nlu_yaml, "safe")
        expected_nlu = rasa.shared.utils.yaml.read_yaml(expected_nlu_yaml, "safe")

        assert actual_nlu == expected_nlu


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
        assistant_name=assistant_name,
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
        assistant_name=assistant_name,
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
def mock_requests_session_post(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("rasa.studio.upload.requests.Session.post", mock)
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
            StudioResult(
                "Upload successful. Request total duration: 0.00 seconds.", True
            ),
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
            StudioResult(
                "Upload successful. Request total duration: 0.00 seconds.", True
            ),
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
    mock_requests_session_post,
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
    mock_requests_session_post.return_value = mock_response

    # Act
    @with_studio_error_handler
    def test_make_request_func():
        return make_request(endpoint, graphql_req)

    result = test_make_request_func()

    # Assert
    assert isinstance(result, StudioResult)
    assert result.message == expected_result.message
    assert result.was_successful == expected_result.was_successful

    mock_requests_session_post.assert_called_once_with(
        endpoint,
        json=graphql_req,
        headers={
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json",
        },
        verify=True,
        timeout=None,
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

    with patch("rasa.studio.upload.requests.Session.post") as mock_post:
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


def test_collect_custom_prompts_all(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Write prompts to the temporary directory
    prompt_names = [
        CONTEXTUAL_RESPONSE_REPHRASER_NAME,
        COMMAND_GENERATOR_NAME,
        ENTERPRISE_SEARCH_NAME,
    ]
    prompts_dict = {prompt_name: prompt_name for prompt_name in prompt_names}
    prompts_dir = tmp_path / DEFAULT_PROMPTS_PATH
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for component_name, prompt_text in prompts_dict.items():
        prompt_file = prompts_dir / f"{component_name}.jinja"
        prompt_file.write_text(prompt_text, encoding="utf-8")

    # Write prompt path to endpoints.yml
    endpoints = {
        "nlg": {
            "prompt": str(prompts_dir / f"{CONTEXTUAL_RESPONSE_REPHRASER_NAME}.jinja")
        }
    }

    # Write prompt paths to config.yml
    config = {
        "pipeline": [
            {
                "name": SingleStepLLMCommandGenerator.__name__,
                "prompt_template": str(prompts_dir / f"{COMMAND_GENERATOR_NAME}.jinja"),
            }
        ],
        "policies": [
            {
                "name": EnterpriseSearchPolicy.__name__,
                "prompt": str(prompts_dir / f"{ENTERPRISE_SEARCH_NAME}.jinja"),
            }
        ],
    }

    # Make sure `collect_custom_prompts` returns the expected prompts
    prompts = rasa.studio.upload.collect_custom_prompts(config, endpoints, tmp_path)
    for prompt_name in prompt_names:
        assert prompt_name in prompts
        assert prompts[prompt_name] == prompt_name


def test_collect_custom_prompts_empty() -> None:
    config: Dict[str, Any] = {"pipeline": [], "policies": []}
    endpoints: Dict[str, Any] = {}
    assert rasa.studio.upload.collect_custom_prompts(config, endpoints) == {}


def test_build_import_request_with_prompts() -> None:
    prompts_json = {CONTEXTUAL_RESPONSE_REPHRASER_NAME: "custom prompt"}
    gql = rasa.studio.upload.build_import_request(
        assistant_name="bot",
        prompts_json=prompts_json,
    )

    assert gql["variables"]["input"]["prompts"] == prompts_json


@pytest.mark.parametrize(
    "data_value, expected_training_paths",
    [
        ("data/path", ["data/path"]),
        (["data/one", "data/two"], ["data/one", "data/two"]),
    ],
)
def test_run_validation_accepts_data_str_or_list(
    monkeypatch, data_value, expected_training_paths
) -> None:
    """Ensure `run_validation` accepts `data` as a string or a list of strings."""
    # Mock TrainingDataImporter
    importer_instance = MagicMock(name="ImporterInstance")
    training_importer_mock = MagicMock()
    training_importer_mock.load_from_dict.return_value = importer_instance
    monkeypatch.setattr(
        rasa.studio.upload, "TrainingDataImporter", training_importer_mock
    )

    # Mock Validator imported inside the function
    validator_instance = MagicMock()
    validator_instance.verify_studio_supported_validations.return_value = True
    ValidatorMock = MagicMock()
    ValidatorMock.from_importer.return_value = validator_instance
    fake_validator_module = SimpleNamespace(Validator=ValidatorMock)
    monkeypatch.setitem(modules, "rasa.validator", fake_validator_module)

    # Prepare args and run
    args = argparse.Namespace(domain="domain.yml", data=data_value, config="config.yml")
    rasa.studio.upload.run_validation(args)

    # Assert TrainingDataImporter receives a list for training_data_paths
    training_importer_mock.load_from_dict.assert_called_once_with(
        domain_path="domain.yml",
        training_data_paths=expected_training_paths,
        config_path="config.yml",
        expand_env_vars=False,
    )

    # Validator is called as expected
    ValidatorMock.from_importer.assert_called_once_with(importer_instance)
    validator_instance.verify_studio_supported_validations.assert_called_once()


def test_get_assistant_id_by_name_returns_id(monkeypatch: MonkeyPatch) -> None:
    """get_assistant_id_by_name returns the id string when the assistant exists."""
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", MagicMock())

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {"assistantByName": {"id": "asst-123", "name": "mybot", "mode": "CALM"}}
    }

    with patch("rasa.studio.upload.requests.Session.post", return_value=mock_response):
        result = get_assistant_id_by_name(
            "mybot", "https://studio.example.com/graphql", verify=True
        )

    assert result == "asst-123"


def test_get_assistant_id_by_name_returns_none_when_not_found(
    monkeypatch: MonkeyPatch,
) -> None:
    """get_assistant_id_by_name returns None when the assistant does not exist."""
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", MagicMock())

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": {"assistantByName": {}}}

    with patch("rasa.studio.upload.requests.Session.post", return_value=mock_response):
        result = get_assistant_id_by_name(
            "nonexistent", "https://studio.example.com/graphql", verify=True
        )

    assert result is None


def test_get_assistant_id_by_name_raises_on_graphql_error(
    monkeypatch: MonkeyPatch,
) -> None:
    """get_assistant_id_by_name raises RasaException when the response has errors."""
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", MagicMock())

    mock_response = MagicMock()
    mock_response.json.return_value = {"errors": [{"message": "Unauthorized"}]}

    with patch("rasa.studio.upload.requests.Session.post", return_value=mock_response):
        with pytest.raises(RasaException, match="Unauthorized"):
            get_assistant_id_by_name(
                "mybot", "https://studio.example.com/graphql", verify=True
            )


def test_check_if_assistant_already_exists_still_works(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression: check_if_assistant_already_exists wrapper returns True/False correctly."""  # noqa: E501
    monkeypatch.setattr(rasa.studio.upload, "KeycloakTokenReader", MagicMock())

    mock_response_exists = MagicMock()
    mock_response_exists.json.return_value = {
        "data": {"assistantByName": {"id": "asst-123", "name": "mybot", "mode": "CALM"}}
    }
    mock_response_not_exists = MagicMock()
    mock_response_not_exists.json.return_value = {"data": {"assistantByName": {}}}

    with patch("rasa.studio.upload.requests.Session.post") as mock_post:
        mock_post.return_value = mock_response_exists
        assert (
            check_if_assistant_already_exists(
                "mybot", "https://studio.example.com/graphql"
            )
            is True
        )

        mock_post.return_value = mock_response_not_exists
        assert (
            check_if_assistant_already_exists(
                "mybot", "https://studio.example.com/graphql"
            )
            is False
        )


def test_build_delete_assistant_request() -> None:
    """build_delete_assistant_request produces the correct GraphQL mutation."""
    result = build_delete_assistant_request("asst-123")

    assert result["variables"]["input"]["assistantId"] == "asst-123"
    assert "DeleteAssistant" in result["query"]
    assert "deleteAssistant" in result["query"]
    assert "DeleteAssistant_AssistantNotFound" in result["query"]


def test_dangerously_delete_existing_flag_deletes_then_uploads(
    monkeypatch: MonkeyPatch,
) -> None:
    """When flag is set and assistant exists, it is deleted then upload proceeds."""
    endpoint = "https://studio.example.com/graphql"
    assistant_name = "mybot"
    assistant_id = "asst-123"

    monkeypatch.setattr(
        rasa.studio.upload,
        "get_assistant_id_by_name",
        MagicMock(return_value=assistant_id),
    )
    mock_delete = MagicMock(return_value=True)
    monkeypatch.setattr(rasa.studio.upload, "delete_assistant", mock_delete)

    args = argparse.Namespace(dangerously_delete_existing=True)
    result = rasa.studio.upload._handle_existing_assistant(
        assistant_name, endpoint, verify=True, args=args
    )

    assert result is True
    mock_delete.assert_called_once_with(assistant_id, assistant_name, endpoint, True)


def test_dangerously_delete_existing_flag_exits_when_delete_fails(
    monkeypatch: MonkeyPatch,
) -> None:
    """When flag is set but delete fails, print_error_and_exit is called."""
    endpoint = "https://studio.example.com/graphql"
    assistant_name = "mybot"

    monkeypatch.setattr(
        rasa.studio.upload,
        "get_assistant_id_by_name",
        MagicMock(return_value="asst-123"),
    )
    monkeypatch.setattr(
        rasa.studio.upload, "delete_assistant", MagicMock(return_value=False)
    )
    mock_exit = MagicMock()
    monkeypatch.setattr(rasa.shared.utils.cli, "print_error_and_exit", mock_exit)
    monkeypatch.setattr(rasa.shared.utils.cli, "print_warning", MagicMock())

    args = argparse.Namespace(dangerously_delete_existing=True)
    rasa.studio.upload._handle_existing_assistant(
        assistant_name, endpoint, verify=True, args=args
    )

    mock_exit.assert_called_once()
    assert "mybot" in mock_exit.call_args[0][0]


def test_dangerously_delete_existing_flag_no_op_when_assistant_does_not_exist(
    monkeypatch: MonkeyPatch,
) -> None:
    """When flag is set but assistant does not exist, upload proceeds without calling delete."""  # noqa: E501
    endpoint = "https://studio.example.com/graphql"

    monkeypatch.setattr(
        rasa.studio.upload,
        "get_assistant_id_by_name",
        MagicMock(return_value=None),
    )
    mock_delete = MagicMock()
    monkeypatch.setattr(rasa.studio.upload, "delete_assistant", mock_delete)

    args = argparse.Namespace(dangerously_delete_existing=True)
    result = rasa.studio.upload._handle_existing_assistant(
        "nonexistent", endpoint, verify=True, args=args
    )

    assert result is True
    mock_delete.assert_not_called()


def test_delete_assistant_success(monkeypatch: MonkeyPatch) -> None:
    """delete_assistant returns True when make_request succeeds."""
    mock_result = StudioResult("Upload successful.", True)
    monkeypatch.setattr(
        rasa.studio.upload, "make_request", MagicMock(return_value=mock_result)
    )

    result = rasa.studio.upload.delete_assistant(
        "asst-123", "mybot", "https://studio.example.com/graphql", verify=True
    )

    assert result is True


def test_delete_assistant_failure(monkeypatch: MonkeyPatch) -> None:
    """delete_assistant returns False when make_request fails."""
    mock_result = StudioResult("Some error.", False)
    monkeypatch.setattr(
        rasa.studio.upload, "make_request", MagicMock(return_value=mock_result)
    )

    result = rasa.studio.upload.delete_assistant(
        "asst-123", "mybot", "https://studio.example.com/graphql", verify=True
    )

    assert result is False


def test_handle_existing_assistant_link_prompt_no(monkeypatch: MonkeyPatch) -> None:
    """When user declines link prompt, print_error_and_exit is called with overwrite hint."""  # noqa: E501
    endpoint = "https://studio.example.com/graphql"
    assistant_name = "mybot"

    monkeypatch.setattr(
        rasa.studio.upload,
        "get_assistant_id_by_name",
        MagicMock(return_value="asst-123"),
    )
    monkeypatch.setattr(
        questionary,
        "confirm",
        MagicMock(return_value=MagicMock(ask=MagicMock(return_value=False))),
    )
    mock_exit = MagicMock()
    monkeypatch.setattr(rasa.shared.utils.cli, "print_error_and_exit", mock_exit)

    args = argparse.Namespace(dangerously_delete_existing=False)
    result = rasa.studio.upload._handle_existing_assistant(
        assistant_name, endpoint, verify=True, args=args
    )

    assert result is False
    mock_exit.assert_called_once()
    assert "--dangerously-delete-existing" in mock_exit.call_args[0][0]


def test_handle_existing_assistant_link_prompt_yes(monkeypatch: MonkeyPatch) -> None:
    """When user accepts link prompt, handle_link is called and upload halts."""
    endpoint = "https://studio.example.com/graphql"
    assistant_name = "mybot"

    monkeypatch.setattr(
        rasa.studio.upload,
        "get_assistant_id_by_name",
        MagicMock(return_value="asst-123"),
    )
    monkeypatch.setattr(
        questionary,
        "confirm",
        MagicMock(return_value=MagicMock(ask=MagicMock(return_value=True))),
    )
    mock_handle_link = MagicMock()
    monkeypatch.setattr("rasa.studio.link.handle_link", mock_handle_link)

    args = argparse.Namespace(dangerously_delete_existing=False, assistant_name=None)
    result = rasa.studio.upload._handle_existing_assistant(
        assistant_name, endpoint, verify=True, args=args
    )

    assert result is False
    mock_handle_link.assert_called_once_with(args)
