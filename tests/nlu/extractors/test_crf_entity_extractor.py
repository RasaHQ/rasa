import copy

from typing import Dict, Text, List, Any

import pytest

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu import registry
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.constants import SPACY_DOCS
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.constants import TEXT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractorGraphComponent


async def test_train_persist_load_with_composite_entities(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext,
):
    importer = RasaFileImporter(
        training_data_paths=["data/test/demo-rasa-composite-entities.yml"]
    )
    training_data = importer.get_nlu_data()

    tokenizer = WhitespaceTokenizer()
    tokenizer.train(training_data)

    crf_extractor = CRFEntityExtractorGraphComponent.create(
        CRFEntityExtractorGraphComponent.get_default_config(),
        default_model_storage,
        Resource("CRFEntityExtractor"),
        default_execution_context,
    )
    crf_extractor.train(training_data)

    message = Message(data={TEXT: "I am looking for an italian restaurant"})

    tokenizer.process(message)
    message2 = copy.deepcopy(message)

    processed_message = crf_extractor.process([message])[0]

    loaded_extractor = CRFEntityExtractorGraphComponent.load(
        CRFEntityExtractorGraphComponent.get_default_config(),
        default_model_storage,
        Resource("CRFEntityExtractor"),
        default_execution_context,
    )

    processed_message2 = loaded_extractor.process([message2])[0]

    assert processed_message2.fingerprint() == processed_message.fingerprint()


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
    config_params: Dict[Text, Any],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    pipeline = [
        {"name": "SpacyNLP", "model": "en_core_web_md"},
        {"name": "SpacyTokenizer"},
    ]

    loaded_pipeline = [
        registry.get_component_class(component.pop("name")).create(
            component, RasaNLUModelConfig()
        )
        for component in copy.deepcopy(pipeline)
    ]

    crf_extractor = CRFEntityExtractorGraphComponent.create(
        {**CRFEntityExtractorGraphComponent.get_default_config(), **config_params},
        default_model_storage,
        Resource("CRFEntityExtractor"),
        default_execution_context,
    )

    importer = RasaFileImporter(training_data_paths=["data/examples/rasa"])
    training_data = importer.get_nlu_data()

    for component in loaded_pipeline:
        component.train(training_data)

    crf_extractor.train(training_data)

    message = Message(data={TEXT: "I am looking for an italian restaurant"})

    for component in loaded_pipeline:
        component.process(message)

    message2 = copy.deepcopy(message)

    processed_message = crf_extractor.process([message])[0]

    loaded_extractor = CRFEntityExtractorGraphComponent.load(
        {**CRFEntityExtractorGraphComponent.get_default_config(), **config_params},
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
    spacy_nlp: Any,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
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
    crf_extractor = CRFEntityExtractorGraphComponent.create(
        {**CRFEntityExtractorGraphComponent.get_default_config(), **component_config},
        default_model_storage,
        Resource("CRFEntityExtractor"),
        default_execution_context,
    )

    spacy_featurizer = SpacyFeaturizer()
    spacy_tokenizer = SpacyTokenizer()

    text = "Rasa is a company in Berlin"
    message = Message(data={TEXT: text})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))

    spacy_tokenizer.process(message)
    spacy_featurizer.process(message)

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
    entity_predictions: List[Dict[Text, float]],
    expected_label: Text,
    expected_confidence: float,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    crf_extractor = CRFEntityExtractorGraphComponent.create(
        CRFEntityExtractorGraphComponent.get_default_config(),
        default_model_storage,
        Resource("CRFEntityExtractor"),
        default_execution_context,
    )

    actual_label, actual_confidence = crf_extractor._most_likely_tag(entity_predictions)

    assert actual_label == expected_label
    assert actual_confidence == expected_confidence
