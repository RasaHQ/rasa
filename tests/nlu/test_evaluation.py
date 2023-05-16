import json
import os
import sys
import textwrap

from pathlib import Path
from typing import Text, List, Dict, Any, Set, Optional

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage

import pytest
from _pytest.monkeypatch import MonkeyPatch
from unittest.mock import Mock, MagicMock

from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.shared.core.trackers import DialogueStateTracker
from tests.conftest import AsyncMock

import rasa.nlu.test
import rasa.shared.nlu.training_data.loading
import rasa.shared.utils.io
import rasa.utils.io
import rasa.model

from rasa.nlu.test import (
    is_token_within_entity,
    do_entities_overlap,
    merge_labels,
    remove_empty_intent_examples,
    remove_empty_response_examples,
    _get_active_entity_extractors,
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
    get_eval_data,
    does_token_cross_borders,
    align_entity_predictions,
    determine_intersection,
    determine_token_labels,
    _remove_entities_of_extractors,
)
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import (
    NO_ENTITY_TAG,
    INTENT,
    INTENT_RANKING_KEY,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    ENTITIES,
)
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    EXTRACTOR,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.model_testing import compare_nlu_models
from rasa.utils.tensorflow.constants import EPOCHS, RUN_EAGERLY

# https://github.com/pytest-dev/pytest-asyncio/issues/68
# this event_loop is used by pytest-asyncio, and redefining it
# is currently the only way of changing the scope of this fixture
from tests.nlu.utilities import write_file_config


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

TRAINING_DATA = rasa.shared.nlu.training_data.loading.load_data(
    "data/test/demo-rasa-more-ents-and-multiplied.yml"
)
NLU_CONFIG = {
    "assistant_id": "placeholder_default",
    "language": "en",
    "pipeline": [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {"name": "LogisticRegressionClassifier"},
    ],
}
N_FOLDS = 2


@pytest.fixture
def mocks_for_test_cross_validate(monkeypatch: MonkeyPatch):
    mock_write_yaml = MagicMock()
    mock_write_yaml.return_value = "write yaml"
    monkeypatch.setattr("rasa.shared.utils.io.write_yaml", mock_write_yaml)

    mock_train_nlu = MagicMock()
    mock_train_nlu.return_value = "train nlu"
    monkeypatch.setattr("rasa.model_training.train_nlu", mock_train_nlu)

    mock_agent_load = MagicMock()
    mock_agent_load.return_value = Agent()
    monkeypatch.setattr("rasa.nlu.test.Agent.load", mock_agent_load)

    mock_RasaYAMLWriter = MagicMock(dump=MagicMock())
    monkeypatch.setattr("rasa.nlu.test.RasaYAMLWriter", mock_RasaYAMLWriter)

    monkeypatch.setattr("rasa.nlu.test.combine_result", AsyncMock())

    mock_evaluate_intents = MagicMock()
    monkeypatch.setattr("rasa.nlu.test.evaluate_intents", mock_evaluate_intents)

    return mock_evaluate_intents


async def mock_combine_result(
    intent_metrics={},
    entity_metrics={},
    response_selection_metrics={},
    processor=None,
    data=None,
    intent_results=[],
    entity_results=None,
    response_selection_results=None,
):
    if intent_results is not None:
        intent_results += IntentEvaluationResult(1, 2, 3, 4)


async def test_cross_validate_evaluate_intents_not_called(
    monkeypatch: MonkeyPatch, mocks_for_test_cross_validate
):
    await cross_validate(
        TRAINING_DATA,
        N_FOLDS,
        NLU_CONFIG,
        successes=False,
        errors=False,
        disable_plotting=True,
        report_as_dict=True,
    )
    mocks_for_test_cross_validate.assert_not_called()


async def test_cross_validate_evaluate_intents_called(
    monkeypatch: MonkeyPatch, mocks_for_test_cross_validate
):
    monkeypatch.setattr(
        "rasa.nlu.test.combine_result", MagicMock(side_effect=mock_combine_result)
    )
    await cross_validate(
        TRAINING_DATA,
        N_FOLDS,
        NLU_CONFIG,
        successes=False,
        errors=False,
        disable_plotting=True,
        report_as_dict=True,
    )

    mocks_for_test_cross_validate.assert_called_once()


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
            {CRFEntityExtractor.__name__},
        )


def test_determine_token_labels_no_extractors():
    label = determine_token_labels(
        CH_correct_segmentation[0], [CH_correct_entity, CH_wrong_entity], None
    )
    assert label == "direction"


def test_determine_token_labels_no_extractors_no_overlap():
    label = determine_token_labels(CH_correct_segmentation[0], EN_targets, None)
    assert label == NO_ENTITY_TAG


def test_determine_token_labels_with_extractors():
    label = determine_token_labels(
        CH_correct_segmentation[0],
        [CH_correct_entity, CH_wrong_entity],
        {SpacyEntityExtractor.__name__, MitieEntityExtractor.__name__},
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
            {"EntityExtractorA"},
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
            {"CRFEntityExtractor"},
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
            {"DIETClassifier"},
            0.87,
        ),
    ],
)
def test_get_entity_confidences(
    token: Token,
    entities: List[Dict[Text, Any]],
    extractors: Set[Text],
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
    td = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.json"
    )
    # include some lookup tables and make sure new td has them
    td = td.merge(TrainingData(lookup_tables=[{"lookup_table": "lookup_entry"}]))
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
    assert clean_td.lookup_tables == td.lookup_tables


@pytest.mark.timeout(
    300, func_only=True
)  # these can take a longer time than the default timeout
async def test_run_evaluation(default_agent: Agent, nlu_as_json_path: Text):
    result = await run_evaluation(
        nlu_as_json_path,
        default_agent.processor,
        errors=False,
        successes=False,
        disable_plotting=True,
    )

    assert result.get("intent_evaluation")
    assert all(
        prediction["confidence"] is not None
        for prediction in result["response_selection_evaluation"]["predictions"]
    )


@pytest.mark.timeout(
    300, func_only=True
)  # these can take a longer time than the default timeout
async def test_run_evaluation_with_regex_message(mood_agent: Agent, tmp_path: Path):
    training_data = textwrap.dedent(
        """
    version: '2.0'
    nlu:
    - intent: goodbye
      examples: |
        - Bye
        - /goodbye{"location": "29432"}
    """
    )

    data_path = tmp_path / "test.yml"
    rasa.shared.utils.io.write_text_file(training_data, data_path)

    # Does not raise
    await run_evaluation(
        str(data_path),
        mood_agent.processor,
        errors=False,
        successes=False,
        disable_plotting=True,
    )


async def test_eval_data(tmp_path: Path, project: Text, trained_rasa_model: Text):
    config_path = os.path.join(project, "config.yml")
    data_importer = TrainingDataImporter.load_nlu_importer_from_config(
        config_path,
        training_data_paths=[
            "data/examples/rasa/demo-rasa.yml",
            "data/examples/rasa/demo-rasa-responses.yml",
        ],
    )

    processor = Agent.load(trained_rasa_model).processor

    data = data_importer.get_nlu_data()
    (intent_results, response_selection_results, entity_results) = await get_eval_data(
        processor, data
    )

    assert len(intent_results) == 46
    assert len(response_selection_results) == 46
    assert len(entity_results) == 46


# FIXME: these tests take too long to run in CI on Windows, disabling them for now
@pytest.mark.skip_on_windows
@pytest.mark.timeout(
    240, func_only=True
)  # these can take a longer time than the default timeout
async def test_run_cv_evaluation():
    td = rasa.shared.nlu.training_data.loading.load_data(
        "data/test/demo-rasa-more-ents-and-multiplied.yml"
    )

    nlu_config = {
        "assistant_id": "placeholder_default",
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {"name": "LogisticRegressionClassifier", EPOCHS: 2},
        ],
    }

    n_folds = 2
    intent_results, entity_results, response_selection_results = await cross_validate(
        td,
        n_folds,
        nlu_config,
        successes=False,
        errors=False,
        disable_plotting=True,
        report_as_dict=True,
    )

    assert len(intent_results.train["Accuracy"]) == n_folds
    assert len(intent_results.train["Precision"]) == n_folds
    assert len(intent_results.train["F1-score"]) == n_folds
    assert len(intent_results.test["Accuracy"]) == n_folds
    assert len(intent_results.test["Precision"]) == n_folds
    assert len(intent_results.test["F1-score"]) == n_folds
    assert all(key in intent_results.evaluation for key in ["errors", "report"])
    assert any(
        isinstance(intent_report, dict)
        and intent_report.get("confused_with") is not None
        for intent_report in intent_results.evaluation["report"].values()
    )
    for extractor_evaluation in entity_results.evaluation.values():
        assert all(key in extractor_evaluation for key in ["errors", "report"])


# FIXME: these tests take too long to run in CI on Windows, disabling them for now
@pytest.mark.skip_on_windows
@pytest.mark.timeout(
    180, func_only=True
)  # these can take a longer time than the default timeout
async def test_run_cv_evaluation_no_entities():
    td = rasa.shared.nlu.training_data.loading.load_data(
        "data/test/demo-rasa-no-ents.yml"
    )

    nlu_config = {
        "assistant_id": "placeholder_default",
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {"name": "LogisticRegressionClassifier", EPOCHS: 25},
        ],
    }

    n_folds = 2
    intent_results, entity_results, response_selection_results = await cross_validate(
        td,
        n_folds,
        nlu_config,
        successes=False,
        errors=False,
        disable_plotting=True,
        report_as_dict=True,
    )

    assert len(intent_results.train["Accuracy"]) == n_folds
    assert len(intent_results.train["Precision"]) == n_folds
    assert len(intent_results.train["F1-score"]) == n_folds
    assert len(intent_results.test["Accuracy"]) == n_folds
    assert len(intent_results.test["Precision"]) == n_folds
    assert len(intent_results.test["F1-score"]) == n_folds
    assert all(key in intent_results.evaluation for key in ["errors", "report"])
    assert any(
        isinstance(intent_report, dict)
        and intent_report.get("confused_with") is not None
        for intent_report in intent_results.evaluation["report"].values()
    )

    assert len(entity_results.train) == 0
    assert len(entity_results.test) == 0
    assert len(entity_results.evaluation) == 0


# FIXME: these tests take too long to run in CI on Windows, disabling them for now
@pytest.mark.skip_on_windows
@pytest.mark.timeout(
    280, func_only=True
)  # these can take a longer time than the default timeout
async def test_run_cv_evaluation_with_response_selector():
    training_data_obj = rasa.shared.nlu.training_data.loading.load_data(
        "data/test/demo-rasa-more-ents-and-multiplied.yml"
    )
    training_data_responses_obj = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.yml"
    )
    training_data_obj = training_data_obj.merge(training_data_responses_obj)

    nlu_config = {
        "assistant_id": "placeholder_default",
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {"name": "LogisticRegressionClassifier", EPOCHS: 25},
            {"name": "CRFEntityExtractor", EPOCHS: 25},
            {"name": "ResponseSelector", EPOCHS: 2, RUN_EAGERLY: True},
        ],
    }

    n_folds = 2
    intent_results, entity_results, response_selection_results = await cross_validate(
        training_data_obj,
        n_folds,
        nlu_config,
        successes=False,
        errors=False,
        disable_plotting=True,
        report_as_dict=True,
    )

    assert len(intent_results.train["Accuracy"]) == n_folds
    assert len(intent_results.train["Precision"]) == n_folds
    assert len(intent_results.train["F1-score"]) == n_folds
    assert len(intent_results.test["Accuracy"]) == n_folds
    assert len(intent_results.test["Precision"]) == n_folds
    assert len(intent_results.test["F1-score"]) == n_folds
    assert all(key in intent_results.evaluation for key in ["errors", "report"])
    assert any(
        isinstance(intent_report, dict)
        and intent_report.get("confused_with") is not None
        for intent_report in intent_results.evaluation["report"].values()
    )

    assert len(response_selection_results.train["Accuracy"]) == n_folds
    assert len(response_selection_results.train["Precision"]) == n_folds
    assert len(response_selection_results.train["F1-score"]) == n_folds
    assert len(response_selection_results.test["Accuracy"]) == n_folds
    assert len(response_selection_results.test["Precision"]) == n_folds
    assert len(response_selection_results.test["F1-score"]) == n_folds
    assert all(
        key in response_selection_results.evaluation for key in ["errors", "report"]
    )

    assert all(
        prediction["confidence"] is not None and prediction["confidence"] != 0.0
        for prediction in response_selection_results.evaluation["predictions"]
    )

    assert any(
        isinstance(intent_report, dict)
        and intent_report.get("confused_with") is not None
        for intent_report in response_selection_results.evaluation["report"].values()
    )

    entity_extractor_name = "CRFEntityExtractor"
    assert len(entity_results.train[entity_extractor_name]["Accuracy"]) == n_folds
    assert len(entity_results.train[entity_extractor_name]["Precision"]) == n_folds
    assert len(entity_results.train[entity_extractor_name]["F1-score"]) == n_folds

    assert len(entity_results.test[entity_extractor_name]["Accuracy"]) == n_folds
    assert len(entity_results.test[entity_extractor_name]["Precision"]) == n_folds
    assert len(entity_results.test[entity_extractor_name]["F1-score"]) == n_folds
    for extractor_evaluation in entity_results.evaluation.values():
        assert all(key in extractor_evaluation for key in ["errors", "report"])


# FIXME: these tests take too long to run in CI on Windows, disabling them for now
@pytest.mark.skip_on_windows
@pytest.mark.timeout(
    280, func_only=True
)  # these can take a longer time than the default timeout
async def test_run_cv_evaluation_lookup_tables():
    td = rasa.shared.nlu.training_data.loading.load_data(
        "data/test/demo-rasa-lookup-ents.yml"
    )

    nlu_config = {
        "assistant_id": "placeholder_default",
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {"name": "LogisticRegressionClassifier", EPOCHS: 1},
            {"name": "RegexEntityExtractor", "use_lookup_tables": True},
        ],
    }

    n_folds = 2
    intent_results, entity_results, response_selection_results = await cross_validate(
        td,
        n_folds,
        nlu_config,
        successes=False,
        errors=False,
        disable_plotting=True,
        report_as_dict=True,
    )

    regex_extractor_name = "RegexEntityExtractor"
    assert regex_extractor_name in entity_results.test

    assert len(entity_results.test[regex_extractor_name]["Accuracy"]) == n_folds
    assert len(entity_results.test[regex_extractor_name]["Precision"]) == n_folds
    assert len(entity_results.test[regex_extractor_name]["F1-score"]) == n_folds

    # All entities in the test set appear in the lookup table,
    # so should get perfect scores
    for fold in range(n_folds):
        assert entity_results.test[regex_extractor_name]["Accuracy"][fold] == 1.0
        assert entity_results.test[regex_extractor_name]["Precision"][fold] == 1.0
        assert entity_results.test[regex_extractor_name]["F1-score"][fold] == 1.0


def test_intent_evaluation_report(tmp_path: Path):
    path = tmp_path / "evaluation"
    path.mkdir()
    report_folder = str(path / "reports")
    report_filename = os.path.join(report_folder, "intent_report.json")

    rasa.shared.utils.io.create_directory(report_folder)

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

    report = json.loads(rasa.shared.utils.io.read_file(report_filename))

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

    assert len(report.keys()) == 5
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

    rasa.shared.utils.io.create_directory(str(report_folder))

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

    report = json.loads(rasa.shared.utils.io.read_file(str(report_filename)))

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

    assert len(report.keys()) == 9
    assert report["A"] == a_results
    assert report["E"] == e_results
    assert report["C"]["confused_with"] == c_confused_with


def test_response_evaluation_report(tmp_path: Path):
    path = tmp_path / "evaluation"
    path.mkdir()
    report_folder = str(path / "reports")
    report_filename = os.path.join(report_folder, "response_selection_report.json")

    rasa.shared.utils.io.create_directory(report_folder)

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

    report = json.loads(rasa.shared.utils.io.read_file(report_filename))

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

    assert len(report.keys()) == 6
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
    "entity_results, expected_extractors",
    [
        ([], set()),
        ([EN_entity_result], {"EntityExtractorA", "EntityExtractorB"}),
        (
            [EN_entity_result, EN_entity_result],
            {"EntityExtractorA", "EntityExtractorB"},
        ),
    ],
)
def test_get_active_entity_extractors(
    entity_results: List[EntityEvaluationResult], expected_extractors: Set[Text]
):
    extractors = _get_active_entity_extractors(entity_results)
    assert extractors == expected_extractors


def test_entity_evaluation_report(tmp_path: Path):

    path = tmp_path / "evaluation"
    path.mkdir()
    report_folder = str(path / "reports")

    report_filename_a = os.path.join(report_folder, "EntityExtractorA_report.json")
    report_filename_b = os.path.join(report_folder, "EntityExtractorB_report.json")

    rasa.shared.utils.io.create_directory(report_folder)

    extractors = _get_active_entity_extractors([EN_entity_result])
    result = evaluate_entities(
        [EN_entity_result],
        extractors,
        report_folder,
        errors=True,
        successes=True,
        disable_plotting=False,
    )

    report_a = json.loads(rasa.shared.utils.io.read_file(report_filename_a))
    report_b = json.loads(rasa.shared.utils.io.read_file(report_filename_b))

    assert len(report_a) == 7
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
        # This happens if response selection test data is present but no response
        # selector is part of the model
        ResponseSelectionEvaluationResult(
            "chitchat/ask_name", None, "What's your name?", None
        ),
    ]
    response_results = remove_empty_response_examples(response_results)

    assert len(response_results) == 2
    assert response_results[0].intent_response_key_target == "chitchat/ask_name"
    assert response_results[0].intent_response_key_prediction == "chitchat/ask_name"
    assert response_results[0].confidence == 0.98765
    assert response_results[0].message == "What's your name?"

    assert response_results[1].intent_response_key_target == "chitchat/ask_name"
    assert response_results[1].intent_response_key_prediction == ""
    assert response_results[1].confidence == 0.0
    assert response_results[1].message == "What's your name?"


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


def test_label_replacement():
    original_labels = ["O", "location"]
    target_labels = ["no_entity", "location"]
    assert substitute_labels(original_labels, "O", "no_entity") == target_labels


async def test_nlu_comparison(
    tmp_path: Path, monkeypatch: MonkeyPatch, nlu_as_json_path: Text
):
    config = {
        "assistant_id": "placeholder_default",
        "language": "en",
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {"name": "KeywordIntentClassifier"},
            {"name": "RegexEntityExtractor"},
        ],
    }
    # the configs need to be at a different path, otherwise the results are
    # combined on the same dictionary key and cannot be plotted properly
    configs = [write_file_config(config).name, write_file_config(config).name]

    monkeypatch.setattr(
        sys.modules["rasa.nlu.test"],
        "get_eval_data",
        AsyncMock(return_value=(1, None, (None,))),
    )
    monkeypatch.setattr(
        sys.modules["rasa.nlu.test"],
        "evaluate_intents",
        Mock(return_value={"f1_score": 1}),
    )

    output = str(tmp_path)
    test_data_importer = TrainingDataImporter.load_from_dict(
        training_data_paths=[nlu_as_json_path]
    )
    test_data = test_data_importer.get_nlu_data()
    await compare_nlu_models(
        configs, test_data, output, runs=2, exclusion_percentages=[50, 80]
    )

    assert set(os.listdir(output)) == {
        "run_1",
        "run_2",
        "results.json",
        "nlu_model_comparison_graph.pdf",
    }

    run_1_path = os.path.join(output, "run_1")
    assert set(os.listdir(run_1_path)) == {"50%_exclusion", "80%_exclusion", "test.yml"}

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
    entity_results: List[EntityEvaluationResult],
    targets: List[Text],
    predictions: List[Text],
    successes: List[Dict[Text, Any]],
    errors: List[Dict[Text, Any]],
):
    actual = collect_successful_entity_predictions(entity_results, targets, predictions)

    assert len(successes) == len(actual)
    assert successes == actual

    actual = collect_incorrect_entity_predictions(entity_results, targets, predictions)

    assert len(errors) == len(actual)
    assert errors == actual


class ConstantProcessor:
    def __init__(self, prediction_to_return: Dict[Text, Any]) -> None:
        self.prediction = prediction_to_return
        self.model_metadata = None

    async def parse_message(
        self,
        message: UserMessage,
        tracker: Optional[DialogueStateTracker] = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        return self.prediction


async def test_replacing_fallback_intent():
    expected_intent = "greet"
    expected_confidence = 0.345
    fallback_prediction = {
        INTENT: {
            INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME,
            PREDICTED_CONFIDENCE_KEY: 1,
        },
        INTENT_RANKING_KEY: [
            {
                INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME,
                PREDICTED_CONFIDENCE_KEY: 1,
            },
            {
                INTENT_NAME_KEY: expected_intent,
                PREDICTED_CONFIDENCE_KEY: expected_confidence,
            },
            {INTENT_NAME_KEY: "some", PREDICTED_CONFIDENCE_KEY: 0.1},
        ],
    }

    processor = ConstantProcessor(fallback_prediction)
    training_data = TrainingData(
        [Message.build("hi", "greet"), Message.build("bye", "bye")]
    )

    intent_evaluations, _, _ = await get_eval_data(processor, training_data)

    assert all(
        prediction.intent_prediction == expected_intent
        and prediction.confidence == expected_confidence
        for prediction in intent_evaluations
    )


async def test_remove_entities_of_extractors():
    extractor = "TestExtractor"
    extractor_2 = "DIET"
    extractor_3 = "YetAnotherExtractor"
    # shouldn't crash when there are no annotations
    _remove_entities_of_extractors({}, [extractor])

    # add some entities
    entities = [
        {
            ENTITY_ATTRIBUTE_TYPE: "time",
            ENTITY_ATTRIBUTE_VALUE: "12:00",
            EXTRACTOR: extractor,
        },
        {
            ENTITY_ATTRIBUTE_TYPE: "location",
            ENTITY_ATTRIBUTE_VALUE: "Berlin - Alexanderplatz",
            EXTRACTOR: extractor_3,
        },
        {
            ENTITY_ATTRIBUTE_TYPE: "name",
            ENTITY_ATTRIBUTE_VALUE: "Joe",
            EXTRACTOR: extractor_2,
        },
    ]
    result_dict = {ENTITIES: entities}
    _remove_entities_of_extractors(result_dict, [extractor, extractor_3])

    assert len(result_dict[ENTITIES]) == 1
    remaining_entity = result_dict[ENTITIES][0]
    assert remaining_entity[EXTRACTOR] == extractor_2
