# coding=utf-8

import logging

import pytest

import rasa.utils.io
from rasa.test import compare_nlu_models
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


def test_determine_token_labels_with_extractors():
    determine_token_labels(
        CH_correct_segmentation[0],
        [CH_correct_entity, CH_wrong_entity],
        [SpacyEntityExtractor.name, MitieEntityExtractor.name],
    )


def test_label_merging():
    aligned_predictions = [
        {
            "target_labels": ["O", "O"],
            "extractor_labels": {"EntityExtractorA": ["O", "O"]},
        },
        {
            "target_labels": ["LOC", "O", "O"],
            "extractor_labels": {"EntityExtractorA": ["O", "O", "O"]},
        },
    ]

    assert all(merge_labels(aligned_predictions) == ["O", "O", "LOC", "O", "O"])
    assert all(
        merge_labels(aligned_predictions, "EntityExtractorA")
        == ["O", "O", "O", "O", "O"]
    )


def test_drop_intents_below_freq():
    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    clean_td = drop_intents_below_freq(td, 0)
    assert clean_td.intents == {"affirm", "goodbye", "greet", "restaurant_search"}

    clean_td = drop_intents_below_freq(td, 10)
    assert clean_td.intents == {"affirm", "restaurant_search"}


def test_run_evaluation(unpacked_trained_moodbot_path):
    data = DEFAULT_DATA_PATH

    result = run_evaluation(
        data, os.path.join(unpacked_trained_moodbot_path, "nlu"), errors=None
    )
    assert result.get("intent_evaluation")
    assert result.get("entity_evaluation").get("CRFEntityExtractor")


def test_run_cv_evaluation():
    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    nlu_config = config.load("sample_configs/config_pretrained_embeddings_spacy.yml")

    n_folds = 2
    intent_results, entity_results = cross_validate(td, n_folds, nlu_config)

    assert len(intent_results.train["Accuracy"]) == n_folds
    assert len(intent_results.train["Precision"]) == n_folds
    assert len(intent_results.train["F1-score"]) == n_folds
    assert len(intent_results.test["Accuracy"]) == n_folds
    assert len(intent_results.test["Precision"]) == n_folds
    assert len(intent_results.test["F1-score"]) == n_folds
    assert len(entity_results.train["CRFEntityExtractor"]["Accuracy"]) == n_folds
    assert len(entity_results.train["CRFEntityExtractor"]["Precision"]) == n_folds
    assert len(entity_results.train["CRFEntityExtractor"]["F1-score"]) == n_folds
    assert len(entity_results.test["CRFEntityExtractor"]["Accuracy"]) == n_folds
    assert len(entity_results.test["CRFEntityExtractor"]["Precision"]) == n_folds
    assert len(entity_results.test["CRFEntityExtractor"]["F1-score"]) == n_folds


def test_intent_evaluation_report(tmpdir_factory):
    path = tmpdir_factory.mktemp("evaluation").strpath
    report_folder = os.path.join(path, "reports")
    report_filename = os.path.join(report_folder, "intent_report.json")

    utils.create_dir(report_folder)

    intent_results = [
        IntentEvaluationResult("", "restaurant_search", "I am hungry", 0.12345),
        IntentEvaluationResult("greet", "greet", "hello", 0.98765),
    ]

    result = evaluate_intents(
        intent_results,
        report_folder,
        successes_filename=None,
        errors_filename=None,
        confmat_filename=None,
        intent_hist_filename=None,
    )

    report = json.loads(rasa.utils.io.read_file(report_filename))

    greet_results = {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1}

    prediction = {
        "text": "hello",
        "intent": "greet",
        "predicted": "greet",
        "confidence": 0.98765,
    }

    assert len(report.keys()) == 4
    assert report["greet"] == greet_results
    assert result["predictions"][0] == prediction


def test_entity_evaluation_report(tmpdir_factory):
    class EntityExtractorA(EntityExtractor):

        provides = ["entities"]

        def __init__(self, component_config=None) -> None:

            super(EntityExtractorA, self).__init__(component_config)

    class EntityExtractorB(EntityExtractor):

        provides = ["entities"]

        def __init__(self, component_config=None) -> None:

            super(EntityExtractorB, self).__init__(component_config)

    path = tmpdir_factory.mktemp("evaluation").strpath
    report_folder = os.path.join(path, "reports")

    report_filename_a = os.path.join(report_folder, "EntityExtractorA_report.json")
    report_filename_b = os.path.join(report_folder, "EntityExtractorB_report.json")

    utils.create_dir(report_folder)
    mock_interpreter = Interpreter(
        [
            EntityExtractorA({"provides": ["entities"]}),
            EntityExtractorB({"provides": ["entities"]}),
        ],
        None,
    )
    extractors = get_entity_extractors(mock_interpreter)
    result = evaluate_entities([EN_entity_result], extractors, report_folder)

    report_a = json.loads(rasa.utils.io.read_file(report_filename_a))
    report_b = json.loads(rasa.utils.io.read_file(report_filename_b))

    assert len(report_a) == 8
    assert report_a["datetime"]["support"] == 1.0
    assert report_b["macro avg"]["recall"] == 0.2
    assert result["EntityExtractorA"]["accuracy"] == 0.75


def test_empty_intent_removal():
    intent_results = [
        IntentEvaluationResult("", "restaurant_search", "I am hungry", 0.12345),
        IntentEvaluationResult("greet", "greet", "hello", 0.98765),
    ]
    intent_results = remove_empty_intent_examples(intent_results)

    assert len(intent_results) == 1
    assert intent_results[0].intent_target == "greet"
    assert intent_results[0].intent_prediction == "greet"
    assert intent_results[0].confidence == 0.98765
    assert intent_results[0].message == "hello"


def test_evaluate_entities_cv_empty_tokens():
    mock_extractors = ["EntityExtractorA", "EntityExtractorB"]
    result = align_entity_predictions(EN_entity_result_no_tokens, mock_extractors)

    assert result == {
        "target_labels": [],
        "extractor_labels": {"EntityExtractorA": [], "EntityExtractorB": []},
    }, "Wrong entity prediction alignment"


def test_evaluate_entities_cv():
    mock_extractors = ["EntityExtractorA", "EntityExtractorB"]
    result = align_entity_predictions(EN_entity_result, mock_extractors)

    assert result == {
        "target_labels": [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "food",
            "location",
            "location",
            "datetime",
        ],
        "extractor_labels": {
            "EntityExtractorA": [
                "O",
                "person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "food",
                "O",
                "location",
                "O",
            ],
            "EntityExtractorB": [
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "movie",
                "movie",
            ],
        },
    }, "Wrong entity prediction alignment"


def test_get_entity_extractors(pretrained_interpreter):
    assert get_entity_extractors(pretrained_interpreter) == {
        "SpacyEntityExtractor",
        "DucklingHTTPExtractor",
    }


def test_remove_pretrained_extractors(pretrained_interpreter):
    target_components_names = ["SpacyNLP"]
    filtered_pipeline = remove_pretrained_extractors(pretrained_interpreter.pipeline)
    filtered_components_names = [c.name for c in filtered_pipeline]
    assert filtered_components_names == target_components_names


def test_label_replacement():
    original_labels = ["O", "location"]
    target_labels = ["no_entity", "location"]
    assert substitute_labels(original_labels, "O", "no_entity") == target_labels


def test_nlu_comparison(tmpdir):
    configs = [
        NLU_DEFAULT_CONFIG_PATH,
        "sample_configs/config_supervised_embeddings.yml",
    ]
    output = tmpdir.strpath

    compare_nlu_models(
        configs, DEFAULT_DATA_PATH, output, runs=2, exclusion_percentages=[50, 80]
    )

    assert set(os.listdir(output)) == {
        "run_1",
        "run_2",
        "results.json",
        "nlu_model_comparison_graph.pdf",
    }

    run_1_path = os.path.join(output, "run_1")
    assert set(os.listdir(run_1_path)) == {"50%_exclusion", "80%_exclusion", "test.md"}
