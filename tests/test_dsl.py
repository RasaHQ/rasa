from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io

from rasa_core.events import SlotSet, ActionExecuted
from rasa_core.training_utils import extract_stories_from_file, \
    extract_story_graph_from_file


def test_can_read_test_story(default_domain):
    stories = extract_stories_from_file("data/dsl_stories/stories.md",
                                        default_domain)
    assert len(stories) == 7
    # this should be the story simple_story_with_only_end -> show_it_all
    # the generated stories are in a non stable order - therefore we need to
    # do some trickery to find the one we want to test
    story = [s for s in stories if len(s.story_steps) == 3][0]
    assert len(story.story_steps) == 3
    assert story.story_steps[0].block_name == 'simple_story_with_only_end'
    assert story.story_steps[1].block_name == 'show_it_all'
    assert story.story_steps[2].block_name == 'show_it_all'

    assert len(story.story_steps[0].events) == 4
    assert story.story_steps[0].events[1] == ActionExecuted("utter_greet")
    assert story.story_steps[0].events[2] == SlotSet("name", "peter")
    assert story.story_steps[0].events[3] == SlotSet("nice_person", "")


def test_persist_and_read_test_story(tmpdir, default_domain):
    graph = extract_story_graph_from_file("data/dsl_stories/stories.md",
                                          default_domain)
    out_path = tmpdir.join("persisted_story.md")
    with io.open(out_path.strpath, "w") as f:
        f.write(graph.as_story_string())

    recovered_stories = extract_stories_from_file(out_path.strpath,
                                                  default_domain)
    existing_stories = {s.as_story_string()
                        for s in graph.build_stories(default_domain)}
    for r in recovered_stories:
        story_str = r.as_story_string()
        assert story_str in existing_stories
        existing_stories.discard(story_str)
