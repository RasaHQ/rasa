import logging

import pytest
from aioresponses import aioresponses
from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError

from rasa.constants import DOMAIN_SCHEMA_FILE, CONFIG_SCHEMA_FILE
from rasa.utils.validation import validate_yaml_schema, InvalidYamlFileError
from rasa.utils.endpoints import EndpointConfig, concat_url
from tests.utilities import latest_request, json_of_latest_request
from rasa.utils.common import sort_list_of_dicts_by_first_key
import rasa.utils.io


async def test_endpoint_config():
    with aioresponses() as mocked:
        endpoint = EndpointConfig(
            "https://example.com/",
            params={"A": "B"},
            headers={"X-Powered-By": "Rasa"},
            basic_auth={"username": "user", "password": "pass"},
            token="mytoken",
            token_name="letoken",
            type="redis",
            port=6379,
            db=0,
            password="password",
            timeout=30000,
        )

        mocked.post(
            "https://example.com/test?A=B&P=1&letoken=mytoken",
            payload={"ok": True},
            repeat=True,
            status=200,
        )

        await endpoint.request(
            "post",
            subpath="test",
            content_type="application/text",
            json={"c": "d"},
            params={"P": "1"},
        )

        r = latest_request(
            mocked, "post", "https://example.com/test?A=B&P=1&letoken=mytoken"
        )

        assert r

        assert json_of_latest_request(r) == {"c": "d"}
        assert r[-1].kwargs.get("params", {}).get("A") == "B"
        assert r[-1].kwargs.get("params", {}).get("P") == "1"
        assert r[-1].kwargs.get("params", {}).get("letoken") == "mytoken"

        # unfortunately, the mock library won't report any headers stored on
        # the session object, so we need to verify them separately
        async with endpoint.session() as s:
            assert s._default_headers.get("X-Powered-By") == "Rasa"
            assert s._default_auth.login == "user"
            assert s._default_auth.password == "pass"


def test_sort_dicts_by_keys():
    test_data = [{"Z": 1}, {"A": 10}]

    expected = [{"A": 10}, {"Z": 1}]
    actual = sort_list_of_dicts_by_first_key(test_data)

    assert actual == expected


@pytest.mark.parametrize(
    "file, schema",
    [
        ("examples/restaurantbot/domain.yml", DOMAIN_SCHEMA_FILE),
        ("sample_configs/config_defaults.yml", CONFIG_SCHEMA_FILE),
        ("sample_configs/config_supervised_embeddings.yml", CONFIG_SCHEMA_FILE),
        ("sample_configs/config_crf_custom_features.yml", CONFIG_SCHEMA_FILE),
    ],
)
def test_validate_yaml_schema(file, schema):
    # should raise no exception
    validate_yaml_schema(rasa.utils.io.read_file(file), schema)


@pytest.mark.parametrize(
    "file, schema",
    [
        ("data/test_domains/invalid_format.yml", DOMAIN_SCHEMA_FILE),
        ("examples/restaurantbot/data/nlu.md", DOMAIN_SCHEMA_FILE),
        ("data/test_config/example_config.yaml", CONFIG_SCHEMA_FILE),
    ],
)
def test_validate_yaml_schema_raise_exception(file, schema):
    with pytest.raises(InvalidYamlFileError):
        validate_yaml_schema(rasa.utils.io.read_file(file), schema)


@pytest.mark.parametrize(
    "base, subpath, expected_result",
    [
        ("https://example.com", None, "https://example.com"),
        ("https://example.com/test", None, "https://example.com/test"),
        ("https://example.com/", None, "https://example.com/"),
        ("https://example.com/", "test", "https://example.com/test"),
        ("https://example.com/", "test/", "https://example.com/test/"),
    ],
)
def test_concat_url(base, subpath, expected_result):
    assert concat_url(base, subpath) == expected_result


def test_warning_for_base_paths_with_trailing_slash(caplog):
    test_path = "base/"

    with caplog.at_level(logging.DEBUG, logger="rasa.utils.endpoints"):
        assert concat_url(test_path, None) == test_path

    assert len(caplog.records) == 1


def test_read_file_with_not_existing_path():
    with pytest.raises(ValueError):
        rasa.utils.io.read_file("some path")


@pytest.mark.parametrize("actual_path", ["", "file.md", "file"])
def test_file_path_validator_with_invalid_paths(actual_path):

    test_error_message = actual_path

    validator = rasa.utils.io.file_type_validator([".yml"], test_error_message)

    document = Document(actual_path)
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == test_error_message


@pytest.mark.parametrize("actual_path", ["domain.yml", "lala.yaml"])
def test_file_path_validator_with_valid_paths(actual_path):

    validator = rasa.utils.io.file_type_validator([".yml", ".yaml"], "error message")

    document = Document(actual_path)
    # If the path is valid there shouldn't be an exception
    assert validator.validate(document) is None


@pytest.mark.parametrize("user_input", ["", "   ", "\t", "\n"])
def test_non_empty_text_validator_with_empty_input(user_input):

    test_error_message = "enter something"

    validator = rasa.utils.io.not_empty_validator(test_error_message)

    document = Document(user_input)
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == test_error_message


@pytest.mark.parametrize("user_input", ["utter_greet", "greet", "Hi there!"])
def test_non_empty_text_validator_with_valid_input(user_input):

    validator = rasa.utils.io.not_empty_validator("error message")

    document = Document(user_input)
    # If there is input there shouldn't be an exception
    assert validator.validate(document) is None


def test_create_validator_from_callable():
    def is_valid(user_input) -> None:
        return user_input == "this passes"

    error_message = "try again"

    validator = rasa.utils.io.create_validator(is_valid, error_message)

    document = Document("this passes")
    assert validator.validate(document) is None

    document = Document("this doesn't")
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == error_message
