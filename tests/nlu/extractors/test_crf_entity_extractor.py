from pathlib import Path

from typing import Dict, Text, List, Any

import pytest

from rasa.nlu.components import ComponentBuilder
from rasa.nlu import train
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.constants import TEXT, SPACY_DOCS, ENTITIES
from rasa.nlu.training_data import Message
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor


def pipeline_from_components(*components: Text) -> List[Dict[Text, Text]]:
    return [{"name": c} for c in components]


async def test_train_persist_load_with_composite_entities(
    component_builder: ComponentBuilder, tmp_path: Path
):
    pipeline = pipeline_from_components("WhitespaceTokenizer", "CRFEntityExtractor")

    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trainer, trained, persisted_path) = await train(
        _config,
        path=str(tmp_path),
        data="data/test/demo-rasa-composite-entities.md",
        component_builder=component_builder,
    )

    assert trainer.pipeline
    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    text = "I am looking for an italian restaurant"
    assert loaded.parse(text) == trained.parse(text)


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
    config_params: Dict[Text, Any], component_builder: ComponentBuilder, tmp_path: Path
):
    pipeline = pipeline_from_components(
        "SpacyNLP", "SpacyTokenizer", "CRFEntityExtractor"
    )
    assert pipeline[2]["name"] == "CRFEntityExtractor"
    pipeline[2].update(config_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trainer, trained, persisted_path) = await train(
        _config,
        path=str(tmp_path),
        data="data/examples/rasa",
        component_builder=component_builder,
    )

    assert trainer.pipeline
    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    text = "I am looking for an italian restaurant"
    assert loaded.parse(text) == trained.parse(text)

    detected_entities = loaded.parse(text).get(ENTITIES)

    assert len(detected_entities) == 1
    assert detected_entities[0]["entity"] == "cuisine"
    assert detected_entities[0]["value"] == "italian"


def test_crf_use_dense_features(spacy_nlp: Any):
    crf_extractor = CRFEntityExtractor(
        component_config={
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
    )

    spacy_featurizer = SpacyFeaturizer()
    spacy_tokenizer = SpacyTokenizer()

    text = "Rasa is a company in Berlin"
    message = Message(text)
    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))

    spacy_tokenizer.process(message)
    spacy_featurizer.process(message)

    text_data = crf_extractor._convert_to_crf_tokens(message)
    features = crf_extractor._crf_tokens_to_features(text_data)

    assert "0:text_dense_features" in features[0]
    dense_features, _ = message.get_dense_features(TEXT, [])

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
):
    crf_extractor = CRFEntityExtractor({"BILOU_flag": True})

    actual_label, actual_confidence = crf_extractor._most_likely_tag(entity_predictions)

    assert actual_label == expected_label
    assert actual_confidence == expected_confidence
