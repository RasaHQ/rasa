from pathlib import Path

from sanic.request import Request
from typing import Text, Iterator, List, Dict, Any

import asyncio

import pytest
from _pytest.tmpdir import TempdirFactory

import rasa.utils.io
from rasa.nlu.constants import NO_ENTITY_TAG
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.test import compare_nlu_models
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.nlu.model import Interpreter, Trainer
from rasa.nlu.test import (
    is_token_within_entity,
    do_entities_overlap,
    merge_labels,
    remove_empty_intent_examples,
    remove_empty_response_examples,
    get_entity_extractors,
    remove_pretrained_extractors,
    drop_intents_below_freq,
    cross_validate,
    run_evaluation,
    substitute_labels,
    IntentEvaluationResult,
    EntityEvaluationResult,
    ResponseSelectionEvaluationResult,
    evaluate_intents,
    evaluate_entities,
    evaluate_response_selections,
    NO_ENTITY,
    collect_successful_entity_predictions,
    collect_incorrect_entity_predictions,
    merge_confidences,
    _get_entity_confidences,
)
from rasa.nlu.test import does_token_cross_borders
from rasa.nlu.test import align_entity_predictions
from rasa.nlu.test import determine_intersection
from rasa.nlu.test import determine_token_labels
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Token
import json
import os
from rasa.nlu import training_data
from tests.nlu.conftest import DEFAULT_DATA_PATH
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.nlu.test import is_response_selector_present
from rasa.utils.tensorflow.constants import EPOCHS, ENTITY_RECOGNITION, RANDOM_SEED

# https://github.com/pytest-dev/pytest-asyncio/issues/68
# this event_loop is used by pytest-asyncio, and redefining it
# is currently the only way of changing the scope of this fixture
from tests.nlu.utilities import write_file_config


@pytest.yield_fixture(scope="session")
def event_loop(request: Request) -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)
    yield loop
    loop.close()


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

EN_entity_result = EntityEvaluationResult(
    EN_targets, EN_predicted, EN_tokens, " ".join([t.text for t in EN_tokens])
)

EN_entity_result_no_tokens = EntityEvaluationResult(EN_targets, EN_predicted, [], "")


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
            [CRFEntityExtractor.name],
        )


def test_determine_token_labels_no_extractors():
    with pytest.raises(ValueError):
        determine_token_labels(
            CH_correct_segmentation[0], [CH_correct_entity, CH_wrong_entity], None
        )


def test_determine_token_labels_no_extractors_no_overlap():
    label = determine_token_labels(CH_correct_segmentation[0], EN_targets, None)
    assert label == NO_ENTITY_TAG


def test_determine_token_labels_with_extractors():
    label = determine_token_labels(
        CH_correct_segmentation[0],
        [CH_correct_entity, CH_wrong_entity],
        [SpacyEntityExtractor.name, MitieEntityExtractor.name],
    )
    assert label == "direction"


@pytest.mark.parametrize(
    "token, entities, extractors, expected_confidence",
    [
        (
            Token("pizza", 4),
            [
                {
                    "start": 4,
                    "end": 9,
                    "value": "pizza",
                    "entity": "food",
                    "extractor": "EntityExtractorA",
                }
            ],
            ["EntityExtractorA"],
            0.0,
        ),
        (Token("pizza", 4), [], ["EntityExtractorA"], 0.0),
        (
            Token("pizza", 4),
            [
                {
                    "start": 4,
                    "end": 9,
                    "value": "pizza",
                    "entity": "food",
                    "confidence_entity": 0.87,
                    "extractor": "CRFEntityExtractor",
                }
            ],
            ["CRFEntityExtractor"],
            0.87,
        ),
        (
            Token("pizza", 4),
            [
                {
                    "start": 4,
                    "end": 9,
                    "value": "pizza",
                    "entity": "food",
                    "confidence_entity": 0.87,
                    "extractor": "DIETClassifier",
                }
            ],
            ["DIETClassifier"],
            0.87,
        ),
    ],
)
def test_get_entity_confidences(
    token: Token,
    entities: List[Dict[Text, Any]],
    extractors: List[Text],
    expected_confidence: float,
):
    confidence = _get_entity_confidences(token, entities, extractors)

    assert confidence == expected_confidence


def test_label_merging():
    import numpy as np

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

    assert np.all(merge_labels(aligned_predictions) == ["O", "O", "LOC", "O", "O"])
    assert np.all(
        merge_labels(aligned_predictions, "EntityExtractorA")
        == ["O", "O", "O", "O", "O"]
    )


def test_confidence_merging():
    import numpy as np

    aligned_predictions = [
        {
            "target_labels": ["O", "O"],
            "extractor_labels": {"EntityExtractorA": ["O", "O"]},
            "confidences": {"EntityExtractorA": [0.0, 0.0]},
        },
        {
            "target_labels": ["LOC", "O", "O"],
            "extractor_labels": {"EntityExtractorA": ["O", "O", "O"]},
            "confidences": {"EntityExtractorA": [0.98, 0.0, 0.0]},
        },
    ]

    assert np.all(
        merge_confidences(aligned_predictions, "EntityExtractorA")
        == [0.0, 0.0, 0.98, 0.0, 0.0]
    )


def test_drop_intents_below_freq():
    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    clean_td = drop_intents_below_freq(td, 0)
    assert clean_td.intents == {
        "affirm",
        "goodbye",
        "greet",
        "restaurant_search",
        "chitchat",
    }

    clean_td = drop_intents_below_freq(td, 10)
    assert clean_td.intents == {"affirm", "restaurant_search"}


def test_run_evaluation(unpacked_trained_moodbot_path):
    result = run_evaluation(
        DEFAULT_DATA_PATH,
        os.path.join(unpacked_trained_moodbot_path, "nlu"),
        errors=False,
        successes=False,
        disable_plotting=True,
    )

    assert result.get("intent_evaluation")


def test_run_cv_evaluation(pretrained_embeddings_spacy_config):
    td = training_data.load_data("data/examples/rasa/demo-rasa.json")

    n_folds = 2
    intent_results, entity_results, response_selection_results = cross_validate(
        td,
        n_folds,
        pretrained_embeddings_spacy_config,
        successes=False,
        errors=False,
        disable_plotting=True,
    )

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


def test_run_cv_evaluation_with_response_selector():
    training_data_obj = training_data.load_data("data/examples/rasa/demo-rasa.md")
    training_data_responses_obj = training_data.load_data(
        "data/examples/rasa/demo-rasa-responses.md"
    )
    training_data_obj = training_data_obj.merge(training_data_responses_obj)

    nlu_config = RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {"name": "DIETClassifier", EPOCHS: 2},
                {"name": "ResponseSelector", EPOCHS: 2},
            ],
        }
    )

    n_folds = 2
    intent_results, entity_results, response_selection_results = cross_validate(
        training_data_obj,
        n_folds,
        nlu_config,
        successes=False,
        errors=False,
        disable_plotting=True,
    )

    assert len(intent_results.train["Accuracy"]) == n_folds
    assert len(intent_results.train["Precision"]) == n_folds
    assert len(intent_results.train["F1-score"]) == n_folds
    assert len(intent_results.test["Accuracy"]) == n_folds
    assert len(intent_results.test["Precision"]) == n_folds
    assert len(intent_results.test["F1-score"]) == n_folds
    assert len(response_selection_results.train["Accuracy"]) == n_folds
    assert len(response_selection_results.train["Precision"]) == n_folds
    assert len(response_selection_results.train["F1-score"]) == n_folds
    assert len(response_selection_results.test["Accuracy"]) == n_folds
    assert len(response_selection_results.test["Precision"]) == n_folds
    assert len(response_selection_results.test["F1-score"]) == n_folds
    assert len(entity_results.train["DIETClassifier"]["Accuracy"]) == n_folds
    assert len(entity_results.train["DIETClassifier"]["Precision"]) == n_folds
    assert len(entity_results.train["DIETClassifier"]["F1-score"]) == n_folds
    assert len(entity_results.test["DIETClassifier"]["Accuracy"]) == n_folds
    assert len(entity_results.test["DIETClassifier"]["Precision"]) == n_folds
    assert len(entity_results.test["DIETClassifier"]["F1-score"]) == n_folds


def test_response_selector_present():
    response_selector_component = ResponseSelector()

    interpreter_with_response_selector = Interpreter(
        [response_selector_component], context=None
    )
    interpreter_without_response_selector = Interpreter([], context=None)

    assert is_response_selector_present(interpreter_with_response_selector)
    assert not is_response_selector_present(interpreter_without_response_selector)


def test_intent_evaluation_report(tmp_path: Path):
    path = tmp_path / "evaluation"
    path.mkdir()
    report_folder = str(path / "reports")
    report_filename = os.path.join(report_folder, "intent_report.json")

    rasa.utils.io.create_directory(report_folder)

    intent_results = [
        IntentEvaluationResult("", "restaurant_search", "I am hungry", 0.12345),
        IntentEvaluationResult("greet", "greet", "hello", 0.98765),
    ]

    result = evaluate_intents(
        intent_results,
        report_folder,
        successes=True,
        errors=True,
        disable_plotting=False,
    )

    report = json.loads(rasa.utils.io.read_file(report_filename))

    greet_results = {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 1,
        "confused_with": {},
    }

    prediction = {
        "text": "hello",
        "intent": "greet",
        "predicted": "greet",
        "confidence": 0.98765,
    }

    assert len(report.keys()) == 4
    assert report["greet"] == greet_results
    assert result["predictions"][0] == prediction

    assert os.path.exists(os.path.join(report_folder, "intent_confusion_matrix.png"))
    assert os.path.exists(os.path.join(report_folder, "intent_histogram.png"))
    assert not os.path.exists(os.path.join(report_folder, "intent_errors.json"))
    assert os.path.exists(os.path.join(report_folder, "intent_successes.json"))


def test_intent_evaluation_report_large(tmp_path: Path):
    path = tmp_path / "evaluation"
    path.mkdir()
    report_folder = path / "reports"
    report_filename = report_folder / "intent_report.json"

    rasa.utils.io.create_directory(str(report_folder))

    def correct(label: Text) -> IntentEvaluationResult:
        return IntentEvaluationResult(label, label, "", 1.0)

    def incorrect(label: Text, _label: Text) -> IntentEvaluationResult:
        return IntentEvaluationResult(label, _label, "", 1.0)

    a_results = [correct("A")] * 10
    b_results = [correct("B")] * 7 + [incorrect("B", "C")] * 3
    c_results = [correct("C")] * 3 + [incorrect("C", "D")] + [incorrect("C", "E")]
    d_results = [correct("D")] * 29 + [incorrect("D", "B")] * 3
    e_results = [incorrect("E", "C")] * 5 + [incorrect("E", "")] * 5

    intent_results = a_results + b_results + c_results + d_results + e_results

    evaluate_intents(
        intent_results,
        str(report_folder),
        successes=False,
        errors=False,
        disable_plotting=True,
    )

    report = json.loads(rasa.utils.io.read_file(str(report_filename)))

    a_results = {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 10,
        "confused_with": {},
    }

    e_results = {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 10,
        "confused_with": {"C": 5, "": 5},
    }

    c_confused_with = {"D": 1, "E": 1}

    assert len(report.keys()) == 8
    assert report["A"] == a_results
    assert report["E"] == e_results
    assert report["C"]["confused_with"] == c_confused_with


def test_response_evaluation_report(tmp_path: Path):
    path = tmp_path / "evaluation"
    path.mkdir()
    report_folder = str(path / "reports")
    report_filename = os.path.join(report_folder, "response_selection_report.json")

    rasa.utils.io.create_directory(report_folder)

    response_results = [
        ResponseSelectionEvaluationResult(
            "chitchat/ask_weather",
            "chitchat/ask_weather",
            "What's the weather",
            0.65432,
        ),
        ResponseSelectionEvaluationResult(
            "chitchat/ask_name", "chitchat/ask_name", "What's your name?", 0.98765
        ),
    ]

    result = evaluate_response_selections(
        response_results,
        report_folder,
        successes=True,
        errors=True,
        disable_plotting=False,
    )

    report = json.loads(rasa.utils.io.read_file(report_filename))

    name_query_results = {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 1,
        "confused_with": {},
    }

    prediction = {
        "text": "What's your name?",
        "intent_response_key_target": "chitchat/ask_name",
        "intent_response_key_prediction": "chitchat/ask_name",
        "confidence": 0.98765,
    }

    assert len(report.keys()) == 5
    assert report["chitchat/ask_name"] == name_query_results
    assert result["predictions"][1] == prediction

    assert os.path.exists(
        os.path.join(report_folder, "response_selection_confusion_matrix.png")
    )
    assert os.path.exists(
        os.path.join(report_folder, "response_selection_histogram.png")
    )
    assert not os.path.exists(
        os.path.join(report_folder, "response_selection_errors.json")
    )
    assert os.path.exists(
        os.path.join(report_folder, "response_selection_successes.json")
    )


@pytest.mark.parametrize(
    "components, expected_extractors",
    [
        ([DIETClassifier({ENTITY_RECOGNITION: False})], set()),
        ([DIETClassifier({ENTITY_RECOGNITION: True})], {"DIETClassifier"}),
        ([CRFEntityExtractor()], {"CRFEntityExtractor"}),
        (
            [SpacyEntityExtractor(), CRFEntityExtractor()],
            {"SpacyEntityExtractor", "CRFEntityExtractor"},
        ),
        ([ResponseSelector()], set()),
    ],
)
def test_get_entity_extractors(components, expected_extractors):
    mock_interpreter = Interpreter(components, None)
    extractors = get_entity_extractors(mock_interpreter)

    assert extractors == expected_extractors


def test_entity_evaluation_report(tmp_path):
    class EntityExtractorA(EntityExtractor):

        provides = ["entities"]

        def __init__(self, component_config=None) -> None:

            super().__init__(component_config)

    class EntityExtractorB(EntityExtractor):

        provides = ["entities"]

        def __init__(self, component_config=None) -> None:

            super().__init__(component_config)

    path = tmp_path / "evaluation"
    path.mkdir()
    report_folder = str(path / "reports")

    report_filename_a = os.path.join(report_folder, "EntityExtractorA_report.json")
    report_filename_b = os.path.join(report_folder, "EntityExtractorB_report.json")

    rasa.utils.io.create_directory(report_folder)
    mock_interpreter = Interpreter(
        [
            EntityExtractorA({"provides": ["entities"]}),
            EntityExtractorB({"provides": ["entities"]}),
        ],
        None,
    )
    extractors = get_entity_extractors(mock_interpreter)
    result = evaluate_entities(
        [EN_entity_result],
        extractors,
        report_folder,
        errors=True,
        successes=True,
        disable_plotting=False,
    )

    report_a = json.loads(rasa.utils.io.read_file(report_filename_a))
    report_b = json.loads(rasa.utils.io.read_file(report_filename_b))

    assert len(report_a) == 6
    assert report_a["datetime"]["support"] == 1.0
    assert report_b["macro avg"]["recall"] == 0.0
    assert report_a["macro avg"]["recall"] == 0.5
    assert result["EntityExtractorA"]["accuracy"] == 0.75

    assert os.path.exists(
        os.path.join(report_folder, "EntityExtractorA_confusion_matrix.png")
    )
    assert os.path.exists(os.path.join(report_folder, "EntityExtractorA_errors.json"))
    assert os.path.exists(
        os.path.join(report_folder, "EntityExtractorA_successes.json")
    )
    assert not os.path.exists(
        os.path.join(report_folder, "EntityExtractorA_histogram.png")
    )


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


def test_empty_response_removal():
    response_results = [
        ResponseSelectionEvaluationResult(None, None, "What's the weather", 0.65432),
        ResponseSelectionEvaluationResult(
            "chitchat/ask_name", "chitchat/ask_name", "What's your name?", 0.98765
        ),
    ]
    response_results = remove_empty_response_examples(response_results)

    assert len(response_results) == 1
    assert response_results[0].intent_response_key_target == "chitchat/ask_name"
    assert response_results[0].intent_response_key_prediction == "chitchat/ask_name"
    assert response_results[0].confidence == 0.98765
    assert response_results[0].message == "What's your name?"


def test_evaluate_entities_cv_empty_tokens():
    mock_extractors = ["EntityExtractorA", "EntityExtractorB"]
    result = align_entity_predictions(EN_entity_result_no_tokens, mock_extractors)

    assert result == {
        "target_labels": [],
        "extractor_labels": {"EntityExtractorA": [], "EntityExtractorB": []},
        "confidences": {"EntityExtractorA": [], "EntityExtractorB": []},
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
        "confidences": {
            "EntityExtractorA": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "EntityExtractorB": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        },
    }, "Wrong entity prediction alignment"


def test_remove_pretrained_extractors(component_builder):
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "SpacyNLP"},
                {"name": "SpacyEntityExtractor"},
                {"name": "DucklingHTTPExtractor"},
            ]
        }
    )
    trainer = Trainer(_config, component_builder)

    target_components_names = ["SpacyNLP"]
    filtered_pipeline = remove_pretrained_extractors(trainer.pipeline)
    filtered_components_names = [c.name for c in filtered_pipeline]
    assert filtered_components_names == target_components_names


def test_label_replacement():
    original_labels = ["O", "location"]
    target_labels = ["no_entity", "location"]
    assert substitute_labels(original_labels, "O", "no_entity") == target_labels


def test_nlu_comparison(tmp_path: Path):
    config = {
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "KeywordIntentClassifier"},
            {"name": "RegexEntityExtractor"},
        ],
    }
    # the configs need to be at a different path, otherwise the results are
    # combined on the same dictionary key and cannot be plotted properly
    configs = [
        write_file_config(config).name,
        write_file_config(config).name,
    ]

    output = str(tmp_path)
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

    exclude_50_path = os.path.join(run_1_path, "50%_exclusion")
    modelnames = [os.path.splitext(os.path.basename(config))[0] for config in configs]

    modeloutputs = set(
        ["train"]
        + [f"{m}_report" for m in modelnames]
        + [f"{m}.tar.gz" for m in modelnames]
    )
    assert set(os.listdir(exclude_50_path)) == modeloutputs


@pytest.mark.parametrize(
    "entity_results,targets,predictions,successes,errors",
    [
        (
            [
                EntityEvaluationResult(
                    entity_targets=[
                        {
                            "start": 17,
                            "end": 24,
                            "value": "Italian",
                            "entity": "cuisine",
                        }
                    ],
                    entity_predictions=[
                        {
                            "start": 17,
                            "end": 24,
                            "value": "Italian",
                            "entity": "cuisine",
                        }
                    ],
                    tokens=[
                        "I",
                        "want",
                        "to",
                        "book",
                        "an",
                        "Italian",
                        "restaurant",
                        ".",
                    ],
                    message="I want to book an Italian restaurant.",
                ),
                EntityEvaluationResult(
                    entity_targets=[
                        {
                            "start": 8,
                            "end": 15,
                            "value": "Mexican",
                            "entity": "cuisine",
                        },
                        {
                            "start": 31,
                            "end": 32,
                            "value": "4",
                            "entity": "number_people",
                        },
                    ],
                    entity_predictions=[],
                    tokens=[
                        "Book",
                        "an",
                        "Mexican",
                        "restaurant",
                        "for",
                        "4",
                        "people",
                        ".",
                    ],
                    message="Book an Mexican restaurant for 4 people.",
                ),
            ],
            [
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                "cuisine",
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                "cuisine",
                NO_ENTITY,
                NO_ENTITY,
                "number_people",
                NO_ENTITY,
                NO_ENTITY,
            ],
            [
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                "cuisine",
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
                NO_ENTITY,
            ],
            [
                {
                    "text": "I want to book an Italian restaurant.",
                    "entities": [
                        {
                            "start": 17,
                            "end": 24,
                            "value": "Italian",
                            "entity": "cuisine",
                        }
                    ],
                    "predicted_entities": [
                        {
                            "start": 17,
                            "end": 24,
                            "value": "Italian",
                            "entity": "cuisine",
                        }
                    ],
                }
            ],
            [
                {
                    "text": "Book an Mexican restaurant for 4 people.",
                    "entities": [
                        {
                            "start": 8,
                            "end": 15,
                            "value": "Mexican",
                            "entity": "cuisine",
                        },
                        {
                            "start": 31,
                            "end": 32,
                            "value": "4",
                            "entity": "number_people",
                        },
                    ],
                    "predicted_entities": [],
                }
            ],
        )
    ],
)
def test_collect_entity_predictions(
    entity_results, targets, predictions, successes, errors
):
    actual = collect_successful_entity_predictions(entity_results, targets, predictions)

    assert len(successes) == len(actual)
    assert successes == actual

    actual = collect_incorrect_entity_predictions(entity_results, targets, predictions)

    assert len(errors) == len(actual)
    assert errors == actual
