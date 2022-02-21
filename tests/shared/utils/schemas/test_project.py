import copy
import textwrap
from typing import Dict, Text, Any

import pytest

import rasa.shared.utils.io
import rasa.shared.utils.validation
from rasa.shared.exceptions import SchemaValidationError

VALID_PROJECT_YAML = textwrap.dedent(
    """
    version: "3.0"
    nlu: ['data/nlu.yml']
    rules: ['data/rules.yml']
    stories: ['data/stories.yml']
    config: 'config.yml'
    domain: ['domain.yml']
    models: 'models/'
    actions: 'actions/'
    test_data: ['tests/']
    train_test_split: 'train_test_split/'
    results: 'results/'
    """
)


@pytest.fixture
def valid_project_json():
    return rasa.shared.utils.io.read_yaml(VALID_PROJECT_YAML)


def test_valid_project_yaml(valid_project_json: Dict[Text, Any]):
    validated = rasa.shared.utils.validation.validate_project_file(valid_project_json)
    assert validated is True


@pytest.mark.parametrize(
    "required_key",
    [
        "version",
        "nlu",
        "rules",
        "stories",
        "config",
        "domain",
        "models",
        "actions",
        "test_data",
        "train_test_split",
        "results",
    ],
)
def test_invalid_project_yaml_missing_required_key(
    valid_project_json: Dict[Text, Any], required_key: Text
):
    invalid_project_json = copy.deepcopy(valid_project_json)
    invalid_project_json.pop(required_key)
    with pytest.raises(SchemaValidationError) as e:
        rasa.shared.utils.validation.validate_project_file(invalid_project_json)

    assert "Failed to validate project.yml, make sure your file is valid." in str(e)


def test_invalid_project_with_new_random_key(valid_project_json: Dict[Text, Any]):
    invalid_project_json = copy.deepcopy(valid_project_json)
    invalid_project_json["random_key"] = "random_value"
    with pytest.raises(SchemaValidationError) as e:
        rasa.shared.utils.validation.validate_project_file(invalid_project_json)

    assert "Additional properties are not allowed ('random_key' was unexpected)" in str(
        e
    )


@pytest.mark.parametrize(
    "key, disallowed_value",
    [
        ("nlu", "data/nlu.yml"),
        ("rules", "data/rules.yml"),
        ("stories", "data/stories.yml"),
        ("config", []),
        ("domain", "domain.yml"),
        ("models", []),
        ("actions", []),
        ("test_data", "tests/"),
        ("train_test_split", []),
        ("results", []),
    ],
)
def test_invalid_project_with_not_allowed_value_types(
    valid_project_json: Dict[Text, Any], key: Text, disallowed_value: Any
):
    invalid_project_json = copy.deepcopy(valid_project_json)
    invalid_project_json[key] = disallowed_value
    with pytest.raises(SchemaValidationError) as e:
        rasa.shared.utils.validation.validate_project_file(invalid_project_json)

    assert "Failed to validate project.yml, make sure your file is valid." in str(e)


def test_valid_project_with_custom_importers(valid_project_json: Dict[Text, Any]):
    valid_project_with_importers = copy.deepcopy(valid_project_json)
    valid_project_with_importers["importers"] = [
        {"name": "module.CustomImporter", "parameter1": "value"}
    ]
    assert (
        rasa.shared.utils.validation.validate_project_file(valid_project_with_importers)
        is True
    )
