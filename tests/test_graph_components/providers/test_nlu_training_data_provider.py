from typing import Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.nlu_training_data_provider import (
    NLUTrainingDataProvider,
)
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.importers.importer import TrainingDataImporter


def test_nlu_training_data_provider_provides_and_persists_data(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config_path: Text,
    nlu_data_path: Text,
):
    resource = Resource("xy")
    importer = TrainingDataImporter.load_from_config(config_path, nlu_data_path)

    config = NLUTrainingDataProvider.get_default_config()
    assert config["language"] == "en"
    assert config["persist"] is False

    provider_1 = NLUTrainingDataProvider(default_model_storage, resource)
    data_from_provider_1 = provider_1.provide({}, importer)

    assert isinstance(provider_1, NLUTrainingDataProvider)
    assert (
        data_from_provider_1.fingerprint()
        == importer.get_nlu_data(config["language"]).fingerprint()
    )

    # provider_2 = NLUTrainingDataProvider(get_default_model_storage, resource)
    # data_from_provider_2 = provider_2.provide({'language': 'en', 'persist': True}, importer)
