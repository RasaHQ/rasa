import os

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


async def test_evaluation_image_creation(tmpdir, default_agent):
    stories_path = os.path.join(tmpdir.strpath, "failed_stories.md")
    img_path = os.path.join(tmpdir.strpath, "story_confmat.pdf")

    await test(
        stories=DEFAULT_STORIES_FILE,
        agent=default_agent,
        out_directory=tmpdir.strpath,
        max_stories=None,
        e2e=False,
    )

    assert os.path.isfile(img_path)
    assert os.path.isfile(stories_path)


async def test_end_to_end_evaluation_script(tmpdir, restaurantbot):
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
        "greet",
        "greet",
        "inform",
        "inform",
        "greet",
        "inform",
        "inform",
        "inform",
        "deny",
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
    assert num_stories == 3


async def test_end_to_end_evaluation_script_unknown_entity(tmpdir, default_agent):
    completed_trackers = await _generate_trackers(
        E2E_STORY_FILE_UNKNOWN_ENTITY, default_agent, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
        completed_trackers, default_agent, use_e2e=True
    )

    assert story_evaluation.evaluation_store.has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 1
    assert num_stories == 1
