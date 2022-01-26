import copy
from typing import Dict, Text, List, Any, Callable

import pytest

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.constants import SPACY_DOCS
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.utils.spacy_utils import SpacyModel, SpacyNLP
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.constants import TEXT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.extractors.crf_entity_extractor import (
    CRFEntityExtractor,
    CRFEntityExtractorOptions,
)


@pytest.fixture()
def crf_entity_extractor(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
) -> Callable[[Dict[Text, Any]], CRFEntityExtractor]:
    def inner(config: Dict[Text, Any]) -> CRFEntityExtractor:
        return CRFEntityExtractor.create(
            {**CRFEntityExtractor.get_default_config(), **config},
            default_model_storage,
            Resource("CRFEntityExtractor"),
            default_execution_context,
        )

    return inner


def test_all_features_defined():
    assert set(CRFEntityExtractorOptions) == set(
        CRFEntityExtractor.function_dict.keys()
    )


async def test_train_persist_load_with_composite_entities(
    crf_entity_extractor: Callable[[Dict[Text, Any]], CRFEntityExtractor],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    whitespace_tokenizer: WhitespaceTokenizer,
):
    importer = RasaFileImporter(
        training_data_paths=["data/test/demo-rasa-composite-entities.yml"]
    )
    training_data = importer.get_nlu_data()

    whitespace_tokenizer.process_training_data(training_data)

    crf_extractor = crf_entity_extractor({})
    crf_extractor.train(training_data)

    message = Message(data={TEXT: "I am looking for an italian restaurant"})

    whitespace_tokenizer.process([message])
    message2 = copy.deepcopy(message)

    processed_message = crf_extractor.process([message])[0]

    loaded_extractor = CRFEntityExtractor.load(
        CRFEntityExtractor.get_default_config(),
        default_model_storage,
        Resource("CRFEntityExtractor"),
        default_execution_context,
    )

    processed_message2 = loaded_extractor.process([message2])[0]

    assert processed_message2.fingerprint() == processed_message.fingerprint()
    assert list(loaded_extractor.entity_taggers.keys()) == list(
        crf_extractor.entity_taggers.keys()
    )


@pytest.mark.parametrize(
    "config_params",
    [
        (
            {
                "features": [
                    ["low", "title", "upper", "pos", "pos2"],
                    [
                        "low",
                        "suffix3",
                        "suffix2",
                        "upper",
                        "title",
                        "digit",
                        "pos",
                        "pos2",
                    ],
                    ["low", "title", "upper", "pos", "pos2"],
                ],
                "BILOU_flag": False,
            }
        ),
        (
            {
                "features": [
                    ["low", "title", "upper", "pos", "pos2"],
                    [
                        "low",
                        "suffix3",
                        "suffix2",
                        "upper",
                        "title",
                        "digit",
                        "pos",
                        "pos2",
                    ],
                    ["low", "title", "upper", "pos", "pos2"],
                ],
                "BILOU_flag": True,
            }
        ),
    ],
)
async def test_train_persist_with_different_configurations(
    crf_entity_extractor: Callable[[Dict[Text, Any]], CRFEntityExtractor],
    config_params: Dict[Text, Any],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    spacy_tokenizer: SpacyTokenizer,
    spacy_featurizer: SpacyFeaturizer,
    spacy_nlp_component: SpacyNLP,
    spacy_model: SpacyModel,
):

    crf_extractor = crf_entity_extractor(config_params)

    importer = RasaFileImporter(training_data_paths=["data/examples/rasa"])
    training_data = importer.get_nlu_data()

    training_data = spacy_nlp_component.process_training_data(
        training_data, spacy_model
    )
    training_data = spacy_tokenizer.process_training_data(training_data)
    training_data = spacy_featurizer.process_training_data(training_data)
    crf_extractor.train(training_data)

    message = Message(data={TEXT: "I am looking for an italian restaurant"})
    messages = spacy_nlp_component.process([message], spacy_model)
    messages = spacy_tokenizer.process(messages)
    message = spacy_featurizer.process(messages)[0]
    message2 = copy.deepcopy(message)

    processed_message = crf_extractor.process([message])[0]

    loaded_extractor = CRFEntityExtractor.load(
        {**CRFEntityExtractor.get_default_config(), **config_params},
        default_model_storage,
        Resource("CRFEntityExtractor"),
        default_execution_context,
    )

    processed_message2 = loaded_extractor.process([message2])[0]

    assert processed_message2.fingerprint() == processed_message.fingerprint()

    detected_entities = processed_message2.get(ENTITIES)

    assert len(detected_entities) == 1
    assert detected_entities[0]["entity"] == "cuisine"
    assert detected_entities[0]["value"] == "italian"


def test_crf_use_dense_features(
    crf_entity_extractor: Callable[[Dict[Text, Any]], CRFEntityExtractor],
    spacy_nlp: Any,
    spacy_featurizer: SpacyFeaturizer,
    spacy_tokenizer: SpacyTokenizer,
):
    component_config = {
        "features": [
            ["low", "title", "upper", "pos", "pos2"],
            [
                "low",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pos",
                "pos2",
                "text_dense_features",
            ],
            ["low", "title", "upper", "pos", "pos2"],
        ]
    }
    crf_extractor = crf_entity_extractor(component_config)

    text = "Rasa is a company in Berlin"
    message = Message(data={TEXT: text})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))

    spacy_tokenizer.process([message])
    spacy_featurizer.process([message])

    text_data = crf_extractor._convert_to_crf_tokens(message)
    features = crf_extractor._crf_tokens_to_features(text_data)

    assert "0:text_dense_features" in features[0]
    dense_features, _ = message.get_dense_features(TEXT, [])
    if dense_features:
        dense_features = dense_features.features

    for i in range(0, len(dense_features[0])):
        assert (
            features[0]["0:text_dense_features"]["text_dense_features"][str(i)]
            == dense_features[0][i]
        )


@pytest.mark.parametrize(
    "entity_predictions, expected_label, expected_confidence",
    [
        ([{"O": 0.34, "B-person": 0.03, "I-person": 0.85}], ["I-person"], [0.88]),
        ([{"O": 0.99, "person": 0.03}], ["O"], [0.99]),
    ],
)
def test_most_likely_entity(
    crf_entity_extractor: Callable[[Dict[Text, Any]], CRFEntityExtractor],
    entity_predictions: List[Dict[Text, float]],
    expected_label: Text,
    expected_confidence: float,
):
    crf_extractor = crf_entity_extractor({"BILOU_flag": True})

    actual_label, actual_confidence = crf_extractor._most_likely_tag(entity_predictions)

    assert actual_label == expected_label
    assert actual_confidence == expected_confidence


def test_process_unfeaturized_input(
    crf_entity_extractor: Callable[[Dict[Text, Any]], CRFEntityExtractor],
):
    crf_extractor = crf_entity_extractor({})
    message_text = "message text"
    message = Message(data={TEXT: message_text})
    processed_message = crf_extractor.process([message])[0]

    assert processed_message.get(TEXT) == message_text
    assert processed_message.get(ENTITIES) == []
