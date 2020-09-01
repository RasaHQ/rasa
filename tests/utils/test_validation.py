import pytest

from jsonschema import ValidationError

from rasa.constants import DOMAIN_SCHEMA_FILE, CONFIG_SCHEMA_FILE

import rasa.utils.validation as validation_utils
import rasa.utils.io as io_utils
import rasa.nlu.schemas.data_schema as schema
from tests.conftest import DEFAULT_NLU_DATA


@pytest.mark.parametrize(
    "file, schema",
    [
        ("examples/moodbot/domain.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_config/config_defaults.yml", CONFIG_SCHEMA_FILE),
        ("data/test_config/config_supervised_embeddings.yml", CONFIG_SCHEMA_FILE),
        ("data/test_config/config_crf_custom_features.yml", CONFIG_SCHEMA_FILE),
    ],
)
def test_validate_yaml_schema(file, schema):
    # should raise no exception
    validation_utils.validate_yaml_schema(io_utils.read_file(file), schema)


@pytest.mark.parametrize(
    "file, schema",
    [
        ("data/test_domains/invalid_format.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_domains/wrong_response_format.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_domains/wrong_custom_response_format.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_domains/empty_response_format.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_config/example_config.yaml", CONFIG_SCHEMA_FILE),
    ],
)
def test_validate_yaml_schema_raise_exception(file, schema):
    with pytest.raises(validation_utils.InvalidYamlFileError):
        validation_utils.validate_yaml_schema(io_utils.read_file(file), schema)


def test_example_training_data_is_valid():
    demo_json = "data/examples/rasa/demo-rasa.json"
    data = io_utils.read_json_file(demo_json)
    validation_utils.validate_training_data(data, schema.rasa_nlu_data_schema())


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"wrong_top_level": []},
        ["this is not a toplevel dict"],
        {
            "rasa_nlu_data": {
                "common_examples": [{"intent": "some example without text"}]
            }
        },
        {
            "rasa_nlu_data": {
                "common_examples": [
                    {
                        "text": "mytext",
                        "entities": [{"start": "INVALID", "end": 0, "entity": "x"}],
                    }
                ]
            }
        },
    ],
)
def test_validate_training_data_is_throwing_exceptions(invalid_data):
    with pytest.raises(ValidationError):
        validation_utils.validate_training_data(
            invalid_data, schema.rasa_nlu_data_schema()
        )


def test_url_data_format():
    data = """
    {
      "rasa_nlu_data": {
        "entity_synonyms": [
          {
            "value": "nyc",
            "synonyms": ["New York City", "nyc", "the big apple"]
          }
        ],
        "common_examples" : [
          {
            "text": "show me flights to New York City",
            "intent": "unk",
            "entities": [
              {
                "entity": "destination",
                "start": 19,
                "end": 32,
                "value": "NYC"
              }
            ]
          }
        ]
      }
    }"""
    fname = io_utils.create_temporary_file(
        data.encode(io_utils.DEFAULT_ENCODING),
        suffix="_tmp_training_data.json",
        mode="w+b",
    )
    data = io_utils.read_json_file(fname)
    assert data is not None
    validation_utils.validate_training_data(data, schema.rasa_nlu_data_schema())


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"group": "a", "role": "c", "value": "text"},
        ["this is not a toplevel dict"],
        {"entity": 1, "role": "c", "value": "text"},
        {"entity": "e", "role": None, "value": "text"},
    ],
)
def test_validate_entity_dict_is_throwing_exceptions(invalid_data):
    with pytest.raises(ValidationError):
        validation_utils.validate_training_data(
            invalid_data, schema.entity_dict_schema()
        )


@pytest.mark.parametrize(
    "data",
    [
        {"entity": "e", "group": "a", "role": "c", "value": "text"},
        {"entity": "e"},
        {"entity": "e", "value": "text"},
        {"entity": "e", "group": "a"},
        {"entity": "e", "role": "c"},
        {"entity": "e", "role": "c", "value": "text"},
        {"entity": "e", "group": "a", "value": "text"},
        {"entity": "e", "group": "a", "role": "c"},
    ],
)
def test_entity_dict_is_valid(data):
    validation_utils.validate_training_data(data, schema.entity_dict_schema())
