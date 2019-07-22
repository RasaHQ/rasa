import pytest

from constants import DOMAIN_SCHEMA_FILE, CONFIG_SCHEMA_FILE

import rasa.utils.validation
import rasa.utils.io


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
    rasa.utils.validation.validate_yaml_schema(rasa.utils.io.read_file(file), schema)


@pytest.mark.parametrize(
    "file, schema",
    [
        ("data/test_domains/invalid_format.yml", DOMAIN_SCHEMA_FILE),
        ("examples/restaurantbot/data/nlu.md", DOMAIN_SCHEMA_FILE),
        ("data/test_config/example_config.yaml", CONFIG_SCHEMA_FILE),
    ],
)
def test_validate_yaml_schema_raise_exception(file, schema):
    with pytest.raises(rasa.utils.validation.InvalidYamlFileError):
        rasa.utils.validation.validate_yaml_schema(
            rasa.utils.io.read_file(file), schema
        )
