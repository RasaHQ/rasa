from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import os

import numpy as np

from rasa_core.training import extract_story_graph
from rasa_core.policies import PolicyTrainer
from rasa_core.events import ActionExecuted, UserUttered
from rasa_core.training.structures import Story


def test_can_read_test_story(default_domain):
    trackers = PolicyTrainer.extract_trackers(
            "data/test_stories/stories.md",
            default_domain)
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
    graph = extract_story_graph("data/test_stories/stories.md",
                                default_domain)
    out_path = tmpdir.join("persisted_story.md")
    with io.open(out_path.strpath, "w") as f:
        f.write(graph.as_story_string())

    recovered_trackers = PolicyTrainer.extract_trackers(out_path.strpath,
                                                        default_domain)
    existing_trackers = PolicyTrainer.extract_trackers(
            "data/test_stories/stories.md",
            default_domain)

    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


def test_persist_and_read_test_story(tmpdir, default_domain):
    graph = extract_story_graph("data/test_stories/stories.md",
                                default_domain)
    out_path = tmpdir.join("persisted_story.md")
    Story(graph.story_steps).dump_to_file(out_path.strpath)

    recovered_trackers = PolicyTrainer.extract_trackers(out_path.strpath,
                                                        default_domain)
    existing_trackers = PolicyTrainer.extract_trackers(
            "data/test_stories/stories.md",
            default_domain)
    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


def test_read_story_file_with_cycles(tmpdir, default_domain):
    graph = extract_story_graph(
                "data/test_stories/stories_with_cycle.md",
                default_domain)

    assert len(graph.story_steps) == 5

    graph_without_cycles = graph.with_cycles_removed()

    assert graph.cyclic_edge_ids != set()
    assert graph_without_cycles.cyclic_edge_ids == set()

    assert len(graph.story_steps) == len(graph_without_cycles.story_steps) == 5

    assert len(graph_without_cycles.story_end_checkpoints) == 2


# TODO creation of training data changed
def test_generate_training_data_with_cycles(tmpdir, default_domain):
    featurizer = BinaryFeaturizer()
    training_data = training.extract_training_data(
            "data/test_stories/stories_with_cycle.md",
            default_domain,
            featurizer,
            augmentation_factor=0,
            max_history=4)

    assert training_data.num_examples() == 15

    np.testing.assert_array_equal(
            training_data.y,
            [2, 4, 0, 2, 4, 0, 1, 0, 2, 4, 0, 1, 0, 0, 3])


def test_visualize_training_data_graph(tmpdir, default_domain):
    graph = extract_story_graph(
                "data/test_stories/stories_with_cycle.md",
                default_domain)

    graph = graph.with_cycles_removed()

    out_path = tmpdir.join("graph.png").strpath

    # this will be the plotted networkx graph
    G = graph.visualize(out_path)

    assert os.path.exists(out_path)

    # we can't check the exact topology - but this should be enough to ensure
    # the visualisation created a sane graph
    assert set(G.nodes()) == set(range(-1, 14))
    assert len(G.edges()) == 16


# TODO creation of training data changed
def test_load_multi_file_training_data(default_domain):
    # the stories file in `data/test_multifile_stories` is the same as in
    # `data/test_stories/stories.md`, but split across multiple files

    data = training.extract_training_data("data/test_stories/stories.md",
                                          default_domain,
                                          featurizer=BinaryFeaturizer(),
                                          max_history=2)

    data_mul = training.extract_training_data("data/test_multifile_stories",
                                              default_domain,
                                              featurizer=BinaryFeaturizer(),
                                              max_history=2)

    assert np.all(data.X == data_mul.X)
    assert np.all(data.y == data_mul.y)


def test_load_training_data_handles_hidden_files(tmpdir, default_domain):
    # create a hidden file

    open(os.path.join(tmpdir.strpath, ".hidden"), 'a').close()
    # create a normal file
    normal_file = os.path.join(tmpdir.strpath, "normal_file")
    open(normal_file, 'a').close()

    data = training.extract_training_data(tmpdir.strpath,
                                          default_domain,
                                          featurizer=BinaryFeaturizer(),
                                          max_history=2)

    assert len(data.X) == 0
    assert len(data.y) == 0
