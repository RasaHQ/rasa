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
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer, \
    BinarySingleStateFeaturizer


def test_can_read_test_story(default_domain):
    # TODO max_history was controlling augmentation
    trackers, _ = PolicyTrainer.extract_trackers(
            "data/test_stories/stories.md",
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False
    )
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

    recovered_trackers, _ = PolicyTrainer.extract_trackers(
            out_path.strpath,
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False
    )
    existing_trackers, _ = PolicyTrainer.extract_trackers(
            "data/test_stories/stories.md",
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False
    )

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

    recovered_trackers, _ = PolicyTrainer.extract_trackers(
            out_path.strpath,
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False
    )
    existing_trackers, _ = PolicyTrainer.extract_trackers(
            "data/test_stories/stories.md",
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False
    )
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


def test_generate_training_data_with_cycles(tmpdir, default_domain):
    featurizer = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                      max_history=4)
    training_trackers, _ = PolicyTrainer.extract_trackers(
        "data/test_stories/stories_with_cycle.md",
        default_domain,
        augmentation_factor=0
    )
    assert len(training_trackers) == 1

    training_data, _ = featurizer.featurize_trackers(training_trackers,
                                                     default_domain)
    y = training_data.y.argmax(axis=-1)
    np.testing.assert_array_equal(
            y,
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


def test_load_multi_file_training_data(default_domain):
    # the stories file in `data/test_multifile_stories` is the same as in
    # `data/test_stories/stories.md`, but split across multiple files
    featurizer = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                      max_history=2)
    trackers, _ = PolicyTrainer.extract_trackers(
        "data/test_stories/stories.md",
        default_domain
    )
    data, _ = featurizer.featurize_trackers(trackers,
                                            default_domain)

    featurizer_mul = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                          max_history=2)
    trackers_mul, _ = PolicyTrainer.extract_trackers(
        "data/test_multifile_stories",
        default_domain
    )
    data_mul, _ = featurizer_mul.featurize_trackers(trackers_mul,
                                                    default_domain)

    # TODO do we want to assert if trackers are the same?
    assert np.all(data.X == data_mul.X)
    assert np.all(data.y == data_mul.y)


def test_load_training_data_handles_hidden_files(tmpdir, default_domain):
    # create a hidden file

    open(os.path.join(tmpdir.strpath, ".hidden"), 'a').close()
    # create a normal file
    normal_file = os.path.join(tmpdir.strpath, "normal_file")
    open(normal_file, 'a').close()

    featurizer = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                      max_history=2)
    trackers, _ = PolicyTrainer.extract_trackers(
        tmpdir.strpath,
        default_domain
    )
    data, _ = featurizer.featurize_trackers(trackers,
                                            default_domain)

    assert len(data.X) == 0
    assert len(data.y) == 0
