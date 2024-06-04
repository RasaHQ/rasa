import os
import textwrap
from pathlib import Path
import json
import logging
from typing import Any, Text, Dict, Callable

import pytest

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.test import (
    _create_data_generator,
    _collect_story_predictions,
    test as evaluate_stories,
    _clean_entity_results,
)
from rasa.core.constants import (
    CONFUSION_MATRIX_STORIES_FILE,
    REPORT_STORIES_FILE,
    FAILED_STORIES_FILE,
    SUCCESSFUL_STORIES_FILE,
    STORIES_WITH_WARNINGS_FILE,
)

# we need this import to ignore the warning...
# noinspection PyUnresolvedReferences
from rasa.nlu.test import evaluate_entities, run_evaluation  # noqa: F401
from rasa.core.agent import Agent, load_agent
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.exceptions import RasaException


@pytest.fixture(scope="module")
async def trained_restaurantbot(trained_async: Callable) -> Path:
    zipped_model = await trained_async(
        domain="data/test_restaurantbot/domain.yml",
        config="data/test_restaurantbot/config.yml",
        training_files=[
            "data/test_restaurantbot/data/rules.yml",
            "data/test_restaurantbot/data/stories.yml",
            "data/test_restaurantbot/data/nlu.yml",
        ],
    )

    if not zipped_model:
        raise RasaException("Model training for formbot failed.")

    return Path(zipped_model)


@pytest.fixture(scope="module")
async def restaurantbot_agent(trained_restaurantbot: Path) -> Agent:
    return await load_agent(str(trained_restaurantbot))


async def test_evaluation_file_creation(
    tmpdir: Path, default_agent: Agent, stories_path: Text
):
    failed_stories_path = str(tmpdir / FAILED_STORIES_FILE)
    success_stories_path = str(tmpdir / SUCCESSFUL_STORIES_FILE)
    stories_with_warnings_path = str(tmpdir / STORIES_WITH_WARNINGS_FILE)
    report_path = str(tmpdir / REPORT_STORIES_FILE)
    confusion_matrix_path = str(tmpdir / CONFUSION_MATRIX_STORIES_FILE)

    await evaluate_stories(
        stories=stories_path,
        agent=default_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=False,
        errors=True,
        successes=True,
        warnings=True,
    )

    assert os.path.isfile(failed_stories_path)
    assert os.path.isfile(success_stories_path)
    assert os.path.isfile(stories_with_warnings_path)
    assert os.path.isfile(report_path)
    assert os.path.isfile(confusion_matrix_path)


async def test_end_to_end_evaluation_script(
    default_agent: Agent, end_to_end_story_path: Text
):
    generator = _create_data_generator(
        end_to_end_story_path, default_agent, use_conversation_test_files=True
    )
    completed_trackers = generator.generate_story_trackers()

    story_evaluation, num_stories, _ = await _collect_story_predictions(
        completed_trackers, default_agent, use_e2e=True
    )

    serialised_store = [
        "utter_greet",
        "action_listen",
        "utter_greet",
        "action_listen",
        "utter_default",
        "action_listen",
        "utter_goodbye",
        "action_listen",
        "utter_greet",
        "action_listen",
        "utter_default",
        "action_listen",
        "greet",
        "greet",
        "default",
        "goodbye",
        "greet",
        "default",
        '[{"name": "Max"}]{"entity": "name", "value": "Max"}',
    ]

    assert story_evaluation.evaluation_store.serialise()[0] == serialised_store
    assert not story_evaluation.evaluation_store.check_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 0
    assert num_stories == 3


async def test_end_to_end_evaluation_script_unknown_entity(
    default_agent: Agent, e2e_story_file_unknown_entity_path: Text
):
    generator = _create_data_generator(
        e2e_story_file_unknown_entity_path,
        default_agent,
        use_conversation_test_files=True,
    )
    completed_trackers = generator.generate_story_trackers()

    story_evaluation, num_stories, _ = await _collect_story_predictions(
        completed_trackers, default_agent
    )

    assert story_evaluation.evaluation_store.check_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 1
    assert num_stories == 1


@pytest.mark.timeout(300, func_only=True)
async def test_end_to_evaluation_with_forms(form_bot_agent: Agent):
    generator = _create_data_generator(
        "data/test_evaluations/test_form_end_to_end_stories.yml",
        form_bot_agent,
        use_conversation_test_files=True,
    )
    test_stories = generator.generate_story_trackers()

    story_evaluation, num_stories, _ = await _collect_story_predictions(
        test_stories, form_bot_agent
    )

    assert not story_evaluation.evaluation_store.check_prediction_target_mismatch()


async def test_source_in_failed_stories(
    tmpdir: Path, default_agent: Agent, e2e_story_file_unknown_entity_path: Text
):
    stories_path = str(tmpdir / FAILED_STORIES_FILE)

    await evaluate_stories(
        stories=e2e_story_file_unknown_entity_path,
        agent=default_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=False,
    )
    story_file_unknown_entity = Path(e2e_story_file_unknown_entity_path).absolute()
    failed_stories = rasa.shared.utils.io.read_file(stories_path)

    assert (
        f"story: simple_story_with_unknown_entity ({story_file_unknown_entity})"
        in failed_stories
    )


async def test_end_to_evaluation_trips_circuit_breaker(
    e2e_story_file_trips_circuit_breaker_path: Text,
    trained_async: Callable,
    tmp_path: Path,
):
    config = textwrap.dedent(
        f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    assistant_id: placeholder_default
    policies:
    - name: MemoizationPolicy
      max_history: 11

    pipeline: []
    """
    )
    config_path = tmp_path / "config.yml"
    rasa.shared.utils.io.write_text_file(config, config_path)

    model_path = await trained_async(
        "data/test_domains/default.yml",
        str(config_path),
        e2e_story_file_trips_circuit_breaker_path,
    )

    agent = await load_agent(model_path)
    generator = _create_data_generator(
        e2e_story_file_trips_circuit_breaker_path,
        agent,
        use_conversation_test_files=True,
    )
    test_stories = generator.generate_story_trackers()

    story_evaluation, num_stories, _ = await _collect_story_predictions(
        test_stories, agent
    )

    circuit_trip_predicted = [
        "utter_greet",
        "utter_greet",
        "utter_greet",
        "utter_greet",
        "utter_greet",
        "utter_greet",
        "utter_greet",
        "utter_greet",
        "utter_greet",
        "utter_greet",
        "circuit breaker tripped",
        "circuit breaker tripped",
    ]

    assert (
        story_evaluation.evaluation_store.action_predictions == circuit_trip_predicted
    )
    assert num_stories == 1


@pytest.mark.parametrize(
    "text, entity, expected_entity",
    [
        (
            "The first one please.",
            {
                "extractor": "DucklingEntityExtractor",
                "entity": "ordinal",
                "confidence": 0.87,
                "start": 4,
                "end": 9,
                "value": 1,
            },
            {
                "text": "The first one please.",
                "entity": "ordinal",
                "start": 4,
                "end": 9,
                "value": "1",
            },
        ),
        (
            "The first one please.",
            {
                "extractor": "CRFEntityExtractor",
                "entity": "ordinal",
                "confidence": 0.87,
                "start": 4,
                "end": 9,
                "value": "1",
            },
            {
                "text": "The first one please.",
                "entity": "ordinal",
                "start": 4,
                "end": 9,
                "value": "1",
            },
        ),
        (
            "Italian food",
            {
                "extractor": "DIETClassifier",
                "entity": "cuisine",
                "confidence": 0.99,
                "start": 0,
                "end": 7,
                "value": "Italian",
            },
            {
                "text": "Italian food",
                "entity": "cuisine",
                "start": 0,
                "end": 7,
                "value": "Italian",
            },
        ),
    ],
)
def test_event_has_proper_implementation(
    text: Text, entity: Dict[Text, Any], expected_entity: Dict[Text, Any]
):
    actual_entities = _clean_entity_results(text, [entity])

    assert actual_entities[0] == expected_entity


@pytest.mark.timeout(600, func_only=True)
@pytest.mark.parametrize(
    "test_file",
    [
        ("data/test_yaml_stories/test_full_retrieval_intent_story.yml"),
        ("data/test_yaml_stories/test_base_retrieval_intent_story.yml"),
    ],
)
async def test_retrieval_intent(response_selector_agent: Agent, test_file: Text):
    generator = _create_data_generator(
        test_file, response_selector_agent, use_conversation_test_files=True
    )
    test_stories = generator.generate_story_trackers()

    story_evaluation, num_stories, _ = await _collect_story_predictions(
        test_stories, response_selector_agent
    )
    # check that test story can either specify base intent or full retrieval intent
    assert not story_evaluation.evaluation_store.check_prediction_target_mismatch()


@pytest.mark.parametrize(
    "test_file",
    [
        ("data/test_yaml_stories/test_full_retrieval_intent_wrong_prediction.yml"),
        ("data/test_yaml_stories/test_base_retrieval_intent_wrong_prediction.yml"),
    ],
)
async def test_retrieval_intent_wrong_prediction(
    tmpdir: Path, response_selector_agent: Agent, test_file: Text
):
    stories_path = str(tmpdir / FAILED_STORIES_FILE)

    await evaluate_stories(
        stories=test_file,
        agent=response_selector_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=True,
    )

    failed_stories = rasa.shared.utils.io.read_file(stories_path)

    # check if the predicted entry contains full retrieval intent
    assert "# predicted: chitchat/ask_name" in failed_stories


# FIXME: these tests take too long to run in the CI, disabling them for now
@pytest.mark.skip_on_ci
@pytest.mark.timeout(240, func_only=True)
async def test_e2e_with_entity_evaluation(e2e_bot_agent: Agent, tmp_path: Path):
    test_file = "data/test_e2ebot/tests/test_stories.yml"

    await evaluate_stories(
        stories=test_file,
        agent=e2e_bot_agent,
        out_directory=str(tmp_path),
        max_stories=None,
        e2e=True,
    )

    report = rasa.shared.utils.io.read_json_file(tmp_path / "TEDPolicy_report.json")
    assert report["name"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 1,
        "confused_with": {},
    }
    assert report["mood"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1-score": 0.6666666666666666,
        "support": 2,
        "confused_with": {},
    }
    errors = rasa.shared.utils.io.read_json_file(tmp_path / "TEDPolicy_errors.json")
    assert len(errors) == 1
    assert errors[0]["text"] == "today I was very cranky"


@pytest.mark.parametrize(
    "stories_yaml,expected_results",
    [
        [
            """
stories:
  - story: story1
    steps:
    - intent: greet
    - action: utter_greet
  - story: story2
    steps:
    - intent: goodbye
    - action: utter_goodbye
  - story: story3
    steps:
    - intent: greet
    - action: utter_greet
    - intent: goodbye
    - action: utter_default
            """,
            {
                "utter_goodbye": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1-score": 1.0,
                    "support": 1,
                },
                "action_listen": {
                    "precision": 1.0,
                    "recall": 0.75,
                    "f1-score": 0.8571428571428571,
                    "support": 4,
                },
                "utter_greet": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1-score": 1.0,
                    "support": 2,
                },
                "utter_default": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1-score": 0.0,
                    "support": 1,
                },
                "accuracy": 0.75,
                "micro avg": {
                    "precision": 1.0,
                    "recall": 0.75,
                    "f1-score": 0.8571428571428571,
                    "support": 8,
                },
                "macro avg": {
                    "precision": 0.75,
                    "recall": 0.6875,
                    "f1-score": 0.7142857142857143,
                    "support": 8,
                },
                "weighted avg": {
                    "precision": 0.875,
                    "recall": 0.75,
                    "f1-score": 0.8035714285714286,
                    "support": 8,
                },
                "conversation_accuracy": {
                    "accuracy": 2.0 / 3.0,
                    "total": 3,
                    "correct": 2,
                    "with_warnings": 0,
                },
            },
        ]
    ],
)
async def test_story_report(
    tmpdir: Path,
    core_agent: Agent,
    stories_yaml: Text,
    expected_results: Dict[Text, Dict[Text, Any]],
) -> None:
    """Check story_report.json file contains correct result keys/values."""

    stories_path = tmpdir / "stories.yml"
    stories_path.write_text(stories_yaml, "utf8")
    out_directory = tmpdir / "results"
    out_directory.mkdir()

    await evaluate_stories(stories_path, core_agent, out_directory=out_directory)
    story_report_path = out_directory / "story_report.json"
    assert story_report_path.exists()

    actual_results = json.loads(story_report_path.read_text("utf8"))
    assert actual_results == expected_results


async def test_story_report_with_empty_stories(tmpdir: Path, core_agent: Agent) -> None:
    stories_path = tmpdir / "stories.yml"
    stories_path.write_text("", "utf8")
    out_directory = tmpdir / "results"
    out_directory.mkdir()

    await evaluate_stories(stories_path, core_agent, out_directory=out_directory)
    story_report_path = out_directory / "story_report.json"
    assert story_report_path.exists()

    actual_results = json.loads(story_report_path.read_text("utf8"))
    assert actual_results == {}


@pytest.mark.parametrize(
    "skip_field,skip_value",
    [
        [None, None],
        ["precision", None],
        ["f1", None],
        ["in_training_data_fraction", None],
        ["report", None],
        ["include_report", False],
    ],
)
async def test_log_evaluation_table(caplog, skip_field, skip_value):
    """Check that _log_evaluation_table correctly omits/includes optional args."""
    arr = [1, 1, 1, 0]
    acc = 0.75
    kwargs = {
        "precision": 0.5,
        "f1": 0.6,
        "in_training_data_fraction": 0.1,
        "report": {"macro f1": 0.7},
    }
    if skip_field:
        kwargs[skip_field] = skip_value
    caplog.set_level(logging.INFO)
    rasa.core.test._log_evaluation_table(arr, "CONVERSATION", acc, **kwargs)

    assert f"Correct:          {int(len(arr) * acc)} / {len(arr)}" in caplog.text
    assert f"Accuracy:         {acc:.3f}" in caplog.text

    if skip_field != "f1":
        assert f"F1-Score:         {kwargs['f1']:5.3f}" in caplog.text
    else:
        assert "F1-Score:" not in caplog.text

    if skip_field != "precision":
        assert f"Precision:        {kwargs['precision']:5.3f}" in caplog.text
    else:
        assert "Precision:" not in caplog.text

    if skip_field != "in_training_data_fraction":
        assert (
            f"In-data fraction: {kwargs['in_training_data_fraction']:.3g}"
            in caplog.text
        )
    else:
        assert "In-data fraction:" not in caplog.text

    if skip_field != "report" and skip_field != "include_report":
        assert f"Classification report: \n{kwargs['report']}" in caplog.text
    else:
        assert "Classification report:" not in caplog.text


@pytest.mark.skip_on_windows
@pytest.mark.parametrize(
    "test_file, correct_intent, correct_entity",
    [
        [
            "data/test_yaml_stories/"
            "test_prediction_with_correct_intent_wrong_entity.yml",
            True,
            False,
        ],
        [
            "data/test_yaml_stories/"
            "test_prediction_with_wrong_intent_correct_entity.yml",
            False,
            True,
        ],
        [
            "data/test_yaml_stories/"
            "test_prediction_with_wrong_intent_wrong_entity.yml",
            False,
            False,
        ],
    ],
)
async def test_wrong_predictions_with_intent_and_entities(
    tmpdir: Path,
    restaurantbot_agent: Agent,
    test_file: Text,
    correct_intent: bool,
    correct_entity: bool,
):
    stories_path = str(tmpdir / FAILED_STORIES_FILE)

    await evaluate_stories(
        stories=test_file,
        agent=restaurantbot_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=True,
    )

    failed_stories = rasa.shared.utils.io.read_file(stories_path)

    if correct_intent and not correct_entity:
        # check if there is no comment on the intent line
        assert "- intent: request_restaurant  # predicted:" not in failed_stories
        # check if there is a comment with the predicted entity on the entity line
        assert "# predicted: cuisine: greek" in failed_stories
        # check that the correctly predicted entity is printed as well
        assert "- seating: outside\n" in failed_stories
        # check that it does not double print entities
        assert failed_stories.count("\n") == 8

    elif not correct_intent and correct_entity:
        # check if there is a comment with the predicted intent on the intent line
        assert "- intent: greet  # predicted: request_restaurant" in failed_stories
        # check if there is no comment on the entity line
        assert "# predicted: cuisine: greek" not in failed_stories
        # check that the correctly predicted entity is printed as well
        assert "- seating: outside\n" in failed_stories
        # check that it does not double print entities
        assert failed_stories.count("\n") == 9

    elif not correct_intent and not correct_entity:
        # check if there is a comment with the predicted intent on the intent line
        assert "- intent: greet  # predicted: request_restaurant" in failed_stories
        # check if there is a comment with the predicted entity on the entity line
        assert "# predicted: cuisine: greek" in failed_stories
        # check that the correctly predicted entity is printed as well
        assert "- seating: outside\n" in failed_stories
        # check that it does not double print entities
        assert failed_stories.count("\n") == 9


@pytest.mark.skip_on_windows
async def test_failed_entity_extraction_comment(
    tmpdir: Path, restaurantbot_agent: Agent
):
    test_file = "data/test_yaml_stories/test_failed_entity_extraction_comment.yml"
    stories_path = str(tmpdir / FAILED_STORIES_FILE)

    await evaluate_stories(
        stories=test_file,
        agent=restaurantbot_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=True,
    )

    failed_stories = rasa.shared.utils.io.read_file(stories_path)
    assert (
        "- intent: request_restaurant"
        "  # predicted: request_restaurant: i am looking for [greek](cuisine) food"
        in failed_stories
    )
