import os
from typing import Text
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.nlu_training_data_provider import (
    NLUTrainingDataProvider,
)
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import (
    DEFAULT_TRAINING_DATA_OUTPUT_PATH,
    TrainingData,
)
from rasa.shared.nlu.training_data.loading import load_data


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
    assert config_1["language"] is None
    assert config_1["persist"] is False

    # create a provider with persist == True
    provider_1 = NLUTrainingDataProvider.create(
        {"language": "en", "persist": True},
        default_model_storage,
        resource,
        default_execution_context,
    )
    assert isinstance(provider_1, NLUTrainingDataProvider)

    # check the data provided is as expected
    data_0 = provider_1.provide(importer)
    data_1 = importer.get_nlu_data(language="en")
    assert data_0.fingerprint() == data_1.fingerprint()

    # check the data was persisted
    with default_model_storage.read_from(resource) as resource_directory:
        data_file = os.path.join(
            str(resource_directory), DEFAULT_TRAINING_DATA_OUTPUT_PATH
        )
        data = load_data(resource_name=data_file, language="en")
        assert os.path.isfile(data_file)
        assert isinstance(data, TrainingData)

        # delete the persisted data
        os.remove(data_file)
        assert not os.path.isfile(data_file)

    # create a provider with persist == False
    provider_2 = NLUTrainingDataProvider.create(
        {"language": "en", "persist": False},
        default_model_storage,
        resource,
        default_execution_context,
    )
    provider_2.provide(importer)

    # check the data was not persisted
    with default_model_storage.read_from(resource) as resource_directory:
        data_file = os.path.join(
            str(resource_directory), DEFAULT_TRAINING_DATA_OUTPUT_PATH
        )
        assert not os.path.isfile(data_file)
