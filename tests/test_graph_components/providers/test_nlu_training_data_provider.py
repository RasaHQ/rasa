from typing import Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.nlu_training_data_provider import (
    NLUTrainingDataProvider,
)
from rasa.shared.importers.importer import TrainingDataImporter


def test_nlu_training_data_provider(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config_path: Text,
    nlu_data_path: Text,
):
    # create a resource and an importer
    resource = Resource("xy")
    importer = TrainingDataImporter.load_from_config(
        config_path=config_path, training_data_paths=[nlu_data_path]
    )

    # check the default configuration is as expected
    config_1 = NLUTrainingDataProvider.get_default_config()
    assert config_1["language"] == "en"
    assert config_1["persist"] is False

    # create a provider without training data
    provider_1 = NLUTrainingDataProvider.create(
        NLUTrainingDataProvider.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )
    assert isinstance(provider_1, NLUTrainingDataProvider)

    # check the data provided is as expected
    data_0 = provider_1.provide({}, importer)
    data_1 = importer.get_nlu_data(config_1["language"])
    assert data_0.fingerprint() == data_1.fingerprint()

    # check persistence has the correct behaviour
    # new config with persist == true
    config_2 = {"language": "en", "persist": True}

    # get data and persist it using config
    data_2 = provider_1.provide(config_2, importer)

    # load the provider
    provider_2 = NLUTrainingDataProvider.load(
        config_2, default_model_storage, resource, default_execution_context
    )
    data_3 = provider_2.provide(config_1)

    assert data_2.fingerprint() != data_3.fingerprint()
