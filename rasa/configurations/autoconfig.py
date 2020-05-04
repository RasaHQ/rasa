import os

import tempfile

from rasa.utils.io import read_yaml_file, write_yaml_file


def create_decent_config():
    import pkg_resources

    default_config_path = pkg_resources.resource_filename(
        __name__, "default_config.yml"
    )
    default_config = read_yaml_file(default_config_path)

    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "config.yml")

    write_yaml_file(default_config, config_path)
    return config_path


# TODO: write content to original config, section should be marked so that we can
#       edit it later on, e.g.:

## *°*°*
## Rasa will try to select a good configuration given your project and data.
## If you want to customize these machine learning parts, just uncomment the below
## configuration and change it as you wish - this will disable the
## automatic configuration.
## Here is the configuration Rasa selected for the most recent training run:
#
# pipeline:
#  - name: SpacyNLP
#  - name: SpacyTokenizer
#  - name: SpacyFeaturizer
#  - name: RegexFeaturizer
#  - name: LexicalSyntacticFeaturizer
#  - name: CountVectorsFeaturizer
#  - name: CountVectorsFeaturizer
#    analyzer: "char_wb"
#    min_ngram: 1
#    max_ngram: 4
#  - name: DIETClassifier
#    epochs: 100
#  - name: EntitySynonymMapper
#  - name: ResponseSelector
#    epochs: 100
#
# policies:
#  - name: MemoizationPolicy
#  - name: TEDPolicy
#    max_history: 5
#    epochs: 100
#  - name: MappingPolicy
#  - name: FormPolicy
## °*°*°
