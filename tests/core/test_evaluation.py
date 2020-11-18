import os
from pathlib import Path
from typing import Any, Text, Dict

import pytest

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.test import (
    _create_data_generator,
    _collect_story_predictions,
    test as evaluate_stories,
    FAILED_STORIES_FILE,
    CONFUSION_MATRIX_STORIES_FILE,
    REPORT_STORIES_FILE,
    SUCCESSFUL_STORIES_FILE,
    _clean_entity_results,
)
from rasa.core.policies.memoization import MemoizationPolicy

# we need this import to ignore the warning...
# noinspection PyUnresolvedReferences
from rasa.nlu.test import run_evaluation
from rasa.core.agent import Agent
from tests.core.conftest import (
    DEFAULT_STORIES_FILE,
    E2E_STORY_FILE_UNKNOWN_ENTITY,
    END_TO_END_STORY_FILE,
    E2E_STORY_FILE_TRIPS_CIRCUIT_BREAKER,
    STORY_FILE_TRIPS_CIRCUIT_BREAKER,
)


async def test_evaluation_file_creation(tmpdir: Path, default_agent: Agent):
    failed_stories_path = str(tmpdir / FAILED_STORIES_FILE)
    success_stories_path = str(tmpdir / SUCCESSFUL_STORIES_FILE)
    report_path = str(tmpdir / REPORT_STORIES_FILE)
    confusion_matrix_path = str(tmpdir / CONFUSION_MATRIX_STORIES_FILE)

    await evaluate_stories(
        stories=DEFAULT_STORIES_FILE,
        agent=default_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=False,
        errors=True,
        successes=True,
    )

    assert os.path.isfile(failed_stories_path)
    assert os.path.isfile(success_stories_path)
    assert os.path.isfile(report_path)
    assert os.path.isfile(confusion_matrix_path)


@pytest.mark.parametrize(
    "test_file", [END_TO_END_STORY_FILE, "data/test_evaluations/end_to_end_story.yml"]
)
async def test_end_to_end_evaluation_script(default_agent: Agent, test_file: Text):
    generator = await _create_data_generator(test_file, default_agent, use_e2e=True)
    completed_trackers = generator.generate_story_trackers()

    story_evaluation, num_stories = await _collect_story_predictions(
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
    assert not story_evaluation.evaluation_store.has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 0
    assert num_stories == 3


async def test_end_to_end_evaluation_script_unknown_entity(default_agent: Agent):
    generator = await _create_data_generator(
        E2E_STORY_FILE_UNKNOWN_ENTITY, default_agent, use_e2e=True
    )
    completed_trackers = generator.generate_story_trackers()

    story_evaluation, num_stories = await _collect_story_predictions(
        completed_trackers, default_agent, use_e2e=True
    )

    assert story_evaluation.evaluation_store.has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 1
    assert num_stories == 1


async def test_end_to_evaluation_with_forms(form_bot_agent: Agent):
    generator = await _create_data_generator(
        "data/test_evaluations/form-end-to-end-stories.md", form_bot_agent, use_e2e=True
    )
    test_stories = generator.generate_story_trackers()

    story_evaluation, num_stories = await _collect_story_predictions(
        test_stories, form_bot_agent, use_e2e=True
    )

    assert not story_evaluation.evaluation_store.has_prediction_target_mismatch()


async def test_source_in_failed_stories(tmpdir: Path, default_agent: Agent):
    stories_path = str(tmpdir / FAILED_STORIES_FILE)

    await evaluate_stories(
        stories=E2E_STORY_FILE_UNKNOWN_ENTITY,
        agent=default_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=False,
    )

    failed_stories = rasa.shared.utils.io.read_file(stories_path)

    assert (
        f"story: simple_story_with_unknown_entity ({E2E_STORY_FILE_UNKNOWN_ENTITY})"
        in failed_stories
    )


async def test_end_to_evaluation_trips_circuit_breaker():
    agent = Agent(
        domain="data/test_domains/default.yml",
        policies=[MemoizationPolicy(max_history=11)],
    )
    training_data = await agent.load_data(STORY_FILE_TRIPS_CIRCUIT_BREAKER)
    agent.train(training_data)

    generator = await _create_data_generator(
        E2E_STORY_FILE_TRIPS_CIRCUIT_BREAKER, agent, use_e2e=True
    )
    test_stories = generator.generate_story_trackers()

    story_evaluation, num_stories = await _collect_story_predictions(
        test_stories, agent, use_e2e=True
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


@pytest.mark.parametrize(
    "test_file",
    [
        ("data/test_yaml_stories/test_full_retrieval_intent_story.yml"),
        ("data/test_yaml_stories/test_base_retrieval_intent_story.yml"),
    ],
)
async def test_retrieval_intent(response_selector_agent: Agent, test_file: Text):
    generator = await _create_data_generator(
        test_file, response_selector_agent, use_e2e=True,
    )
    test_stories = generator.generate_story_trackers()

    story_evaluation, num_stories = await _collect_story_predictions(
        test_stories, response_selector_agent, use_e2e=True
    )
    # check that test story can either specify base intent or full retrieval intent
    assert not story_evaluation.evaluation_store.has_prediction_target_mismatch()


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
