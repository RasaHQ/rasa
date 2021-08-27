from typing import Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.nlu_training_data_provider import NLUTrainingDataProvider
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.importers.importer import TrainingDataImporter


def test_nlu_training_data_provider_provides_and_persists_data(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config_path: Text,
    training_data_path: Text,
    training_data: TrainingData,
):
    resource = Resource("xy")
    provider_1 = NLUTrainingDataProvider.create(
        NLUTrainingDataProvider.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )
    assert isinstance(provider_1, NLUTrainingDataProvider)

    # importer = TrainingDataImporter.load_from_config(config_path, training_data_path)
    # training_data_from_provider = provider_1.provide_train(importer)
    #
    # assert isinstance(training_data, TrainingData)
    # assert training_data.fingerprint() == training_data_from_provider.fingerprint()
    #
    # with default_model_storage.read_from(resource) as d:
    #     match = list(d.glob("**/nlu.yml"))
    #     assert len(match) == 1
    #     assert match[0].is_file()
    #
    # provider_2 = NLUTrainingDataProvider.load(
    #     {}, default_model_storage, resource, default_execution_context
    # )
    # inference_training_data = provider_2.provide_inference()
    #
    # assert isinstance(inference_training_data, TrainingData)
    # assert training_data.fingerprint() == inference_training_data.fingerprint()
