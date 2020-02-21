import json
import tempfile
from typing import Text

import pytest

from rasa.nlu import config
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.registry import registered_pipeline_templates
from rasa.nlu.model import Trainer
from tests.nlu.utilities import write_file_config


def test_blank_config(default_config):
    file_config = {}
    f = write_file_config(file_config)
    final_config = config.load(f.name)
    assert final_config.as_dict() == default_config.as_dict()


def test_invalid_config_json():
    file_config = """pipeline: [pretrained_embeddings_spacy"""  # invalid yaml
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(file_config)
        f.flush()
        with pytest.raises(config.InvalidConfigError):
            config.load(f.name)


def test_invalid_pipeline_template():
    args = {"pipeline": "my_made_up_name"}
    f = write_file_config(args)
    with pytest.raises(config.InvalidConfigError) as execinfo:
        config.load(f.name)
    assert "unknown pipeline template" in str(execinfo.value)


def test_invalid_many_tokenizers_in_config():
    nlu_config = {
        "pipeline": [{"name": "WhitespaceTokenizer"}, {"name": "SpacyTokenizer"}],
    }

    with pytest.raises(config.InvalidConfigError) as execinfo:
        Trainer(config.RasaNLUModelConfig(nlu_config))
    assert "More then one tokenizer is used" in str(execinfo.value)


def test_invalid_requred_components_in_config():
    spacy_config = {
        "pipeline": [{"name": "WhitespaceTokenizer"}, {"name": "SpacyFeaturizer"}],
    }
    convert_config = {
        "pipeline": [{"name": "WhitespaceTokenizer"}, {"name": "ConveRTFeaturizer"}],
    }
    lm_config = {
        "pipeline": [
            {"name": "ConveRTTokenizer"},
            {"name": "LanguageModelFeaturizer"},
        ],
    }
    count_vectors_config = {
        "pipeline": [{"name": "CountVectorsFeaturizer"}],
    }

    with pytest.raises(config.InvalidConfigError) as execinfo:
        Trainer(config.RasaNLUModelConfig(spacy_config))
    assert "Add required components to the pipeline" in str(execinfo.value)

    with pytest.raises(config.InvalidConfigError) as execinfo:
        Trainer(config.RasaNLUModelConfig(convert_config))
    assert "Add required components to the pipeline" in str(execinfo.value)

    with pytest.raises(config.InvalidConfigError) as execinfo:
        Trainer(config.RasaNLUModelConfig(lm_config))
    assert "Add required components to the pipeline" in str(execinfo.value)

    with pytest.raises(config.InvalidConfigError) as execinfo:
        Trainer(config.RasaNLUModelConfig(count_vectors_config))
    assert "Add required components to the pipeline" in str(execinfo.value)


@pytest.mark.parametrize(
    "pipeline_template", list(registered_pipeline_templates.keys())
)
def test_pipeline_registry_lookup(pipeline_template: Text):
    args = {"pipeline": pipeline_template}
    f = write_file_config(args)
    final_config = config.load(f.name)
    components = [c for c in final_config.pipeline]

    assert json.dumps(components, sort_keys=True) == json.dumps(
        registered_pipeline_templates[pipeline_template], sort_keys=True
    )


def test_default_config_file():
    final_config = config.RasaNLUModelConfig()
    assert len(final_config) > 1


def test_set_attr_on_component(pretrained_embeddings_spacy_config):
    idx_classifier = pretrained_embeddings_spacy_config.component_names.index(
        "SklearnIntentClassifier"
    )
    idx_tokenizer = pretrained_embeddings_spacy_config.component_names.index(
        "SpacyTokenizer"
    )
    pretrained_embeddings_spacy_config.set_component_attr(idx_classifier, C=324)

    assert pretrained_embeddings_spacy_config.for_component(idx_tokenizer) == {
        "name": "SpacyTokenizer"
    }
    assert pretrained_embeddings_spacy_config.for_component(idx_classifier) == {
        "name": "SklearnIntentClassifier",
        "C": 324,
    }


def test_override_defaults_supervised_embeddings_pipeline(supervised_embeddings_config):
    builder = ComponentBuilder()

    idx_featurizer = supervised_embeddings_config.component_names.index(
        "CountVectorsFeaturizer"
    )
    idx_classifier = supervised_embeddings_config.component_names.index(
        "EmbeddingIntentClassifier"
    )

    config_featurizer = supervised_embeddings_config.for_component(idx_featurizer)
    config_classifier = supervised_embeddings_config.for_component(idx_classifier)

    component1 = builder.create_component(
        config_featurizer, supervised_embeddings_config
    )
    assert component1.max_ngram == 1

    component2 = builder.create_component(
        config_classifier, supervised_embeddings_config
    )
    assert component2.component_config["epochs"] == 3
