import os
from typing import Text

from pathlib import Path

from rasa.core.test import _generate_trackers, collect_story_predictions, test

# we need this import to ignore the warning...
# noinspection PyUnresolvedReferences
from rasa.nlu.test import run_evaluation
from rasa.core.agent import Agent
from tests.core.conftest import (
    DEFAULT_STORIES_FILE,
    E2E_STORY_FILE_UNKNOWN_ENTITY,
    END_TO_END_STORY_FILE,
)


async def test_evaluation_image_creation(tmpdir: Path, default_agent: Agent):
    stories_path = str(tmpdir / "failed_stories.md")
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


async def test_end_to_end_evaluation_script(restaurantbot: Text):
    restaurantbot = Agent.load(restaurantbot)
    completed_trackers = await _generate_trackers(
        END_TO_END_STORY_FILE, restaurantbot, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
        completed_trackers, restaurantbot, use_e2e=True
    )

    serialised_store = [
        "utter_ask_howcanhelp",
        "action_listen",
        "utter_ask_howcanhelp",
        "action_listen",
        "utter_on_it",
        "utter_ask_cuisine",
        "action_listen",
        "utter_ask_numpeople",
        "action_listen",
        "utter_ask_howcanhelp",
        "action_listen",
        "utter_on_it",
        "utter_ask_numpeople",
        "action_listen",
        "utter_ask_moreupdates",
        "action_listen",
        "utter_ask_moreupdates",
        "action_listen",
        "utter_ack_dosearch",
        "action_search_restaurants",
        "action_suggest",
        "action_listen",
        "utter_ask_howcanhelp",
        "action_listen",
        "greet",
        "greet",
        "inform",
        "inform",
        "greet",
        "inform",
        "inform",
        "inform",
        "deny",
        "greet",
        "[moderately](price:moderate)",
        "[east](location)",
        "[french](cuisine)",
        "[cheap](price:lo)",
        "[french](cuisine)",
        "[bombay](location)",
        "[six](people:6)",
        "[moderately](price:moderate)",
    ]

    assert story_evaluation.evaluation_store.serialise()[0] == serialised_store
    assert not story_evaluation.evaluation_store.has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 0
    assert num_stories == 4


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


async def test_end_to_evaluation_with_forms(form_bot: Agent):
    test_stories = await _generate_trackers(
        "examples/formbot/tests/end-to-end-stories.md", form_bot, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
        test_stories, form_bot, use_e2e=True
    )

    assert not story_evaluation.evaluation_store.has_prediction_target_mismatch()
