import os
from pathlib import Path

import rasa.utils.io
from rasa.core.test import (
    _generate_trackers,
    collect_story_predictions,
    test,
    FAILED_STORIES_FILE,
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


async def test_evaluation_image_creation(tmpdir: Path, default_agent: Agent):
    stories_path = str(tmpdir / FAILED_STORIES_FILE)
    img_path = str(tmpdir / "story_confmat.pdf")

    await test(
        stories=DEFAULT_STORIES_FILE,
        agent=default_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=False,
    )

    assert os.path.isfile(img_path)
    assert os.path.isfile(stories_path)


async def test_end_to_end_evaluation_script(default_agent: Agent):
    completed_trackers = await _generate_trackers(
        END_TO_END_STORY_FILE, default_agent, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
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
    completed_trackers = await _generate_trackers(
        E2E_STORY_FILE_UNKNOWN_ENTITY, default_agent, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
        completed_trackers, default_agent, use_e2e=True
    )

    assert story_evaluation.evaluation_store.has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 1
    assert num_stories == 1


async def test_end_to_evaluation_with_forms(form_bot_agent: Agent):
    test_stories = await _generate_trackers(
        "data/test_evaluations/form-end-to-end-stories.md", form_bot_agent, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
        test_stories, form_bot_agent, use_e2e=True
    )

    assert not story_evaluation.evaluation_store.has_prediction_target_mismatch()


async def test_source_in_failed_stories(tmpdir: Path, default_agent: Agent):
    stories_path = str(tmpdir / FAILED_STORIES_FILE)

    await test(
        stories=E2E_STORY_FILE_UNKNOWN_ENTITY,
        agent=default_agent,
        out_directory=str(tmpdir),
        max_stories=None,
        e2e=False,
    )

    failed_stories = rasa.utils.io.read_file(stories_path)

    assert (
        f"## simple_story_with_unknown_entity ({E2E_STORY_FILE_UNKNOWN_ENTITY})"
        in failed_stories
    )


async def test_end_to_evaluation_trips_circuit_breaker():
    agent = Agent(
        domain="data/test_domains/default.yml",
        policies=[MemoizationPolicy(max_history=11)],
    )
    training_data = await agent.load_data(STORY_FILE_TRIPS_CIRCUIT_BREAKER)
    agent.train(training_data)

    test_stories = await _generate_trackers(
        E2E_STORY_FILE_TRIPS_CIRCUIT_BREAKER, agent, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
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
