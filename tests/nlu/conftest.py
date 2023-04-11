import copy
from typing import Any, Callable, Dict, List, Text, Tuple, Type, Union

import pytest

from rasa.engine.graph import ExecutionContext, GraphComponent, GraphSchema
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.constants import EPOCHS, RANDOM_SEED


@pytest.fixture()
def pretrained_embeddings_spacy_config() -> Dict:
    return {
        "assistant_id": "placeholder_default",
        "language": "en",
        "pipeline": [
            {"name": "SpacyNLP", "model": "en_core_web_md"},
            {"name": "SpacyTokenizer"},
            {"name": "SpacyFeaturizer"},
            {"name": "RegexFeaturizer"},
            {"name": "CRFEntityExtractor", EPOCHS: 1, RANDOM_SEED: 42},
            {"name": "EntitySynonymMapper"},
            {"name": "SklearnIntentClassifier"},
        ],
    }


@pytest.fixture()
def train_and_preprocess(
    default_model_storage: ModelStorage,
) -> Callable[..., Tuple[TrainingData, List[GraphComponent]]]:
    def inner(
        pipeline: List[Dict[Text, Any]], training_data: Union[Text, TrainingData]
    ) -> Tuple[TrainingData, List[GraphComponent]]:

        if isinstance(training_data, str):
            importer = RasaFileImporter(training_data_paths=[training_data])
            training_data: TrainingData = importer.get_nlu_data()

        def create_component(
            component_class: Type[GraphComponent], config: Dict[Text, Any], idx: int
        ) -> GraphComponent:
            node_name = f"{component_class.__name__}_{idx}"
            execution_context = ExecutionContext(GraphSchema({}), node_name=node_name)
            resource = Resource(node_name)
            return component_class.create(
                {**component_class.get_default_config(), **config},
                default_model_storage,
                resource,
                execution_context,
            )

        component_pipeline = [
            create_component(component.pop("component"), component, idx)
            for idx, component in enumerate(copy.deepcopy(pipeline))
        ]

        for component in component_pipeline:
            if hasattr(component, "train"):
                component.train(training_data)
            if hasattr(component, "process_training_data"):
                component.process_training_data(training_data)

        return training_data, component_pipeline

    return inner


@pytest.fixture()
def process_message(default_model_storage: ModelStorage) -> Callable[..., Message]:
    def inner(loaded_pipeline: List[GraphComponent], message: Message) -> Message:

        for component in loaded_pipeline:
            component.process([message])

        return message

    return inner


@pytest.fixture()
def spacy_tokenizer() -> SpacyTokenizer:
    return SpacyTokenizer(SpacyTokenizer.get_default_config())


@pytest.fixture()
def spacy_featurizer() -> SpacyFeaturizer:
    return SpacyFeaturizer(SpacyFeaturizer.get_default_config(), name="SpacyFeaturizer")


@pytest.fixture()
def mitie_tokenizer() -> MitieTokenizer:
    return MitieTokenizer(MitieTokenizer.get_default_config())
