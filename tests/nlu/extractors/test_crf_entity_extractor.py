from typing import Dict, Text

import pytest

from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.constants import TEXT, SPACY_DOCS, ENTITIES
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor


def test_crf_extractor(spacy_nlp):
    examples = [
        Message(
            "anywhere in the west",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 16, "end": 20, "value": "west", "entity": "location"}
                ],
                SPACY_DOCS[TEXT]: spacy_nlp("anywhere in the west"),
            },
        ),
        Message(
            "central indian restaurant",
            {
                "intent": "restaurant_search",
                "entities": [
                    {
                        "start": 0,
                        "end": 7,
                        "value": "central",
                        "entity": "location",
                        "extractor": "random_extractor",
                    },
                    {
                        "start": 8,
                        "end": 14,
                        "value": "indian",
                        "entity": "cuisine",
                        "extractor": "CRFEntityExtractor",
                    },
                ],
                SPACY_DOCS[TEXT]: spacy_nlp("central indian restaurant"),
            },
        ),
    ]

    extractor = CRFEntityExtractor(
        component_config={
            "features": [
                ["low", "title", "upper", "pos", "pos2"],
                ["low", "suffix3", "suffix2", "upper", "title", "digit", "pos", "pos2"],
                ["low", "title", "upper", "pos", "pos2"],
            ]
        }
    )
    tokenizer = SpacyTokenizer()

    training_data = TrainingData(training_examples=examples)
    tokenizer.train(training_data)
    extractor.train(training_data)

    sentence = "italian restaurant"
    message = Message(sentence, {SPACY_DOCS[TEXT]: spacy_nlp(sentence)})

    tokenizer.process(message)
    extractor.process(message)

    detected_entities = message.get(ENTITIES)

    assert len(detected_entities) == 1
    assert detected_entities[0]["entity"] == "cuisine"
    assert detected_entities[0]["value"] == "italian"


def test_crf_json_from_BILOU(spacy_nlp):
    ext = CRFEntityExtractor(
        component_config={
            "features": [
                ["low", "title", "upper", "pos", "pos2"],
                [
                    "low",
                    "bias",
                    "suffix3",
                    "suffix2",
                    "upper",
                    "title",
                    "digit",
                    "pos",
                    "pos2",
                ],
                ["low", "title", "upper", "pos", "pos2"],
            ]
        }
    )

    sentence = "I need a home cleaning close-by"

    message = Message(sentence, {SPACY_DOCS[TEXT]: spacy_nlp(sentence)})

    tokenizer = SpacyTokenizer()
    tokenizer.process(message)

    r = ext._tag_confidences(
        message,
        [
            {"O": 1.0},
            {"O": 1.0},
            {"O": 1.0},
            {"B-what": 1.0},
            {"L-what": 1.0},
            {"B-where": 1.0},
            {"I-where": 1.0},
            {"L-where": 1.0},
        ],
    )
    assert len(r) == 2, "There should be two entities"

    assert r[0]["confidence"]  # confidence should exist
    del r[0]["confidence"]
    assert r[0] == {"start": 9, "end": 22, "value": "home cleaning", "entity": "what"}

    assert r[1]["confidence"]  # confidence should exist
    del r[1]["confidence"]
    assert r[1] == {"start": 23, "end": 31, "value": "close-by", "entity": "where"}


def test_crf_json_from_non_BILOU(spacy_nlp):
    from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor

    ext = CRFEntityExtractor(
        component_config={
            "BILOU_flag": False,
            "features": [
                ["low", "title", "upper", "pos", "pos2"],
                ["low", "suffix3", "suffix2", "upper", "title", "digit", "pos", "pos2"],
                ["low", "title", "upper", "pos", "pos2"],
            ],
        }
    )
    sentence = "I need a home cleaning close-by"

    message = Message(sentence, {SPACY_DOCS[TEXT]: spacy_nlp(sentence)})

    tokenizer = SpacyTokenizer()
    tokenizer.process(message)

    rs = ext._tag_confidences(
        message,
        [
            {"O": 1.0},
            {"O": 1.0},
            {"O": 1.0},
            {"what": 1.0},
            {"what": 1.0},
            {"where": 1.0},
            {"where": 1.0},
            {"where": 1.0},
        ],
    )

    assert len(rs) == 2

    for r in rs:
        assert r["confidence"]  # confidence should exist
        del r["confidence"]

    assert rs[0] == {"start": 9, "end": 22, "value": "home cleaning", "entity": "what"}
    assert rs[1] == {"start": 23, "end": 31, "value": "close-by", "entity": "where"}


def test_crf_use_dense_features(spacy_nlp):
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
    features = crf_extractor._sentence_to_features(text_data)

    assert "0:text_dense_features" in features[0]
    for i in range(0, len(message.data.get("text_dense_features")[0])):
        assert (
            features[0]["0:text_dense_features"]["text_dense_features"][str(i)]
            == message.data.get("text_dense_features")[0][i]
        )


@pytest.mark.parametrize(
    "entity_predictions, expected_label, expected_confidence",
    [
        ({"O": 0.34, "B-person": 0.03, "I-person": 0.85}, "I-person", 0.88),
        ({"O": 0.99, "person": 0.03}, "O", 0.99),
        (None, "", 0.0),
    ],
)
def test__most_likely_entity(
    entity_predictions: Dict[Text, float],
    expected_label: Text,
    expected_confidence: float,
):
    crf_extractor = CRFEntityExtractor({"BILOU_flag": True})

    actual_label, actual_confidence = crf_extractor._most_likely_tag(entity_predictions)

    assert actual_label == expected_label
    assert actual_confidence == expected_confidence
