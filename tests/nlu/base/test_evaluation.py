# coding=utf-8

import logging

import pytest

import rasa.utils.io
from rasa.test import test_compare_nlu
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.nlu.model import Interpreter
from rasa.nlu.test import (
    is_token_within_entity,
    do_entities_overlap,
    merge_labels,
    remove_empty_intent_examples,
    get_entity_extractors,
    remove_pretrained_extractors,
    drop_intents_below_freq,
    cross_validate,
    run_evaluation,
    substitute_labels,
    IntentEvaluationResult,
    EntityEvaluationResult,
    evaluate_intents,
    evaluate_entities,
)
from rasa.nlu.test import does_token_cross_borders
from rasa.nlu.test import align_entity_predictions
from rasa.nlu.test import determine_intersection
from rasa.nlu.test import determine_token_labels
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token
from rasa.nlu import utils
import json
import os
from rasa.nlu import training_data, config
from tests.nlu import utilities
from tests.nlu.conftest import DEFAULT_DATA_PATH, NLU_DEFAULT_CONFIG_PATH

logging.basicConfig(level="DEBUG")

CONFIG_FOLDERS_PATH = './sample_configs'


@pytest.fixture(scope="session")
def pretrained_interpreter(component_builder, tmpdir_factory):
    conf = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "SpacyNLP"},
                {"name": "SpacyEntityExtractor"},
                {"name": "DucklingHTTPExtractor"},
            ]
        }
    )
    return utilities.interpreter_for(
        component_builder,
        data="./data/examples/rasa/demo-rasa.json",
        path=tmpdir_factory.mktemp("projects").strpath,
        config=conf,
    )


# Chinese Example
# "对面食过敏" -> To be allergic to wheat-based food
CH_wrong_segmentation = [
    Token("对面", 0),
    Token("食", 2),
    Token("过敏", 3),  # opposite, food, allergy
]
CH_correct_segmentation = [
    Token("对", 0),
    Token("面食", 1),
    Token("过敏", 3),  # towards, wheat-based food, allergy
]
CH_wrong_entity = {"start": 0, "end": 2, "value": "对面", "entity": "direction"}
CH_correct_entity = {"start": 1, "end": 3, "value": "面食", "entity": "food_type"}

# EN example
# "Hey Robot, I would like to eat pizza near Alexanderplatz tonight"
EN_indices = [0, 4, 9, 11, 13, 19, 24, 27, 31, 37, 42, 57]
EN_tokens = [
    "Hey",
    "Robot",
    ",",
    "I",
    "would",
    "like",
    "to",
    "eat",
    "pizza",
    "near",
    "Alexanderplatz",
    "tonight",
]
EN_tokens = [Token(t, i) for t, i in zip(EN_tokens, EN_indices)]

EN_targets = [
    {"start": 31, "end": 36, "value": "pizza", "entity": "food"},
    {"start": 37, "end": 56, "value": "near Alexanderplatz", "entity": "location"},
    {"start": 57, "end": 64, "value": "tonight", "entity": "datetime"},
]

EN_predicted = [
    {
        "start": 4,
        "end": 9,
        "value": "Robot",
        "entity": "person",
        "extractor": "EntityExtractorA",
    },
    {
        "start": 31,
        "end": 36,
        "value": "pizza",
        "entity": "food",
        "extractor": "EntityExtractorA",
    },
    {
        "start": 42,
        "end": 56,
        "value": "Alexanderplatz",
        "entity": "location",
        "extractor": "EntityExtractorA",
    },
    {
        "start": 42,
        "end": 64,
        "value": "Alexanderplatz tonight",
        "entity": "movie",
        "extractor": "EntityExtractorB",
    },
]

EN_entity_result = EntityEvaluationResult(EN_targets, EN_predicted, EN_tokens)

EN_entity_result_no_tokens = EntityEvaluationResult(EN_targets, EN_predicted, [])


def test_token_entity_intersection():
    # included
    intsec = determine_intersection(CH_correct_segmentation[1], CH_correct_entity)
    assert intsec == len(CH_correct_segmentation[1].text)

    # completely outside
    intsec = determine_intersection(CH_correct_segmentation[2], CH_correct_entity)
    assert intsec == 0

    # border crossing
    intsec = determine_intersection(CH_correct_segmentation[1], CH_wrong_entity)
    assert intsec == 1


def test_token_entity_boundaries():
    # smaller and included
    assert is_token_within_entity(CH_wrong_segmentation[1], CH_correct_entity)
    assert not does_token_cross_borders(CH_wrong_segmentation[1], CH_correct_entity)

    # exact match
    assert is_token_within_entity(CH_correct_segmentation[1], CH_correct_entity)
    assert not does_token_cross_borders(CH_correct_segmentation[1], CH_correct_entity)

    # completely outside
    assert not is_token_within_entity(CH_correct_segmentation[0], CH_correct_entity)
    assert not does_token_cross_borders(CH_correct_segmentation[0], CH_correct_entity)

    # border crossing
    assert not is_token_within_entity(CH_wrong_segmentation[0], CH_correct_entity)
    assert does_token_cross_borders(CH_wrong_segmentation[0], CH_correct_entity)


def test_entity_overlap():
    assert do_entities_overlap([CH_correct_entity, CH_wrong_entity])
    assert not do_entities_overlap(EN_targets)


def test_determine_token_labels_throws_error():
    with pytest.raises(ValueError):
        determine_token_labels(
            CH_correct_segmentation[0],
            [CH_correct_entity, CH_wrong_entity],
            ["CRFEntityExtractor"],
        )


def test_determine_token_labels_no_extractors():
    with pytest.raises(ValueError):
        determine_token_labels(
            CH_correct_segmentation[0], [CH_correct_entity, CH_wrong_entity], None
        )


def test_determine_token_labels_no_extractors_no_overlap():
    determine_token_labels(CH_correct_segmentation[0], EN_targets, None)


def test_nlu_comparison(tmpdir):

    configs = [NLU_DEFAULT_CONFIG_PATH, "sample_configs/config_supervised_embeddings.yml"]
    output = tmpdir.strpath

    test_compare_nlu(configs,
                     DEFAULT_DATA_PATH,
                     output,
                     runs=2,
                     exclusion_percentages=[80, 95])

    assert os.listdir(output) == ["run_1", "run_2", "results.json", "nlu_model_comparison_graph.pdf"]

    run_1_path = os.path.join(output, "run_1")
    assert os.listdir(run_1_path) == ["80%_exclusion", "95%_exclusion", "test.md"]

