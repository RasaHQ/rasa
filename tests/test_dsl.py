from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io

from rasa_core.events import SlotSet, ActionExecuted, UserUttered
from rasa_core.featurizers import BinaryFeaturizer
from rasa_core.training import (
    extract_trackers_from_file,
    extract_story_graph_from_file)
from rasa_core.training.structures import Story


def test_can_read_test_story(default_domain):
    trackers = extract_trackers_from_file("data/test_stories/stories.md",
                                          default_domain,
                                          featurizer=BinaryFeaturizer())
    assert len(trackers) == 7
    # this should be the story simple_story_with_only_end -> show_it_all
    # the generated stories are in a non stable order - therefore we need to
    # do some trickery to find the one we want to test
    tracker = [t for t in trackers if len(t.events) == 5][0]
    assert tracker.events[0] == ActionExecuted("action_listen")
    assert tracker.events[1] == UserUttered(
            "simple",
            intent={"name": "simple", "confidence": 1.0},
            parse_data={'text': 'simple',
                        'intent_ranking': [{'confidence': 1.0,
                                            'name': 'simple'}],
                        'intent': {'confidence': 1.0, 'name': 'simple'},
                        'entities': []})
    assert tracker.events[2] == ActionExecuted("utter_default")
    assert tracker.events[3] == ActionExecuted("utter_greet")
    assert tracker.events[4] == ActionExecuted("action_listen")


def test_persist_and_read_test_story_graph(tmpdir, default_domain):
    graph = extract_story_graph_from_file("data/test_stories/stories.md",
                                          default_domain)
    out_path = tmpdir.join("persisted_story.md")
    with io.open(out_path.strpath, "w") as f:
        f.write(graph.as_story_string())

    recovered_trackers = extract_trackers_from_file(out_path.strpath,
                                                    default_domain,
                                                    BinaryFeaturizer())
    existing_trackers = extract_trackers_from_file(
        "data/test_stories/stories.md",
        default_domain,
        BinaryFeaturizer())

    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


def test_persist_and_read_test_story(tmpdir, default_domain):
    graph = extract_story_graph_from_file("data/test_stories/stories.md",
                                          default_domain)
    out_path = tmpdir.join("persisted_story.md")
    Story(graph.story_steps).dump_to_file(out_path.strpath)

    recovered_trackers = extract_trackers_from_file(out_path.strpath,
                                                    default_domain,
                                                    BinaryFeaturizer())
    existing_trackers = extract_trackers_from_file(
            "data/test_stories/stories.md",
            default_domain,
            BinaryFeaturizer())
    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)
