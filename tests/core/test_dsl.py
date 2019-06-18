import io
import os

import json
from collections import Counter

import numpy as np

from rasa.core import training
from rasa.core.events import ActionExecuted, UserUttered
from rasa.core.training.structures import Story
from rasa.core.featurizers import (
    MaxHistoryTrackerFeaturizer,
    BinarySingleStateFeaturizer,
)


async def test_can_read_test_story(default_domain):
    trackers = await training.load_data(
        "data/test_stories/stories.md",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
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
        parse_data={
            "text": "/simple",
            "intent_ranking": [{"confidence": 1.0, "name": "simple"}],
            "intent": {"confidence": 1.0, "name": "simple"},
            "entities": [],
        },
    )
    assert tracker.events[2] == ActionExecuted("utter_default")
    assert tracker.events[3] == ActionExecuted("utter_greet")
    assert tracker.events[4] == ActionExecuted("action_listen")


async def test_can_read_test_story_with_checkpoint_after_or(default_domain):
    trackers = await training.load_data(
        "data/test_stories/stories_checkpoint_after_or.md",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    # there should be only 2 trackers
    assert len(trackers) == 2


async def test_persist_and_read_test_story_graph(tmpdir, default_domain):
    graph = await training.extract_story_graph(
        "data/test_stories/stories.md", default_domain
    )
    out_path = tmpdir.join("persisted_story.md")
    with open(out_path.strpath, "w", encoding="utf-8") as f:
        f.write(graph.as_story_string())

    recovered_trackers = await training.load_data(
        out_path.strpath,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_trackers = await training.load_data(
        "data/test_stories/stories.md",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )

    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


async def test_persist_and_read_test_story(tmpdir, default_domain):
    graph = await training.extract_story_graph(
        "data/test_stories/stories.md", default_domain
    )
    out_path = tmpdir.join("persisted_story.md")
    Story(graph.story_steps).dump_to_file(out_path.strpath)

    recovered_trackers = await training.load_data(
        out_path.strpath,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_trackers = await training.load_data(
        "data/test_stories/stories.md",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


async def test_read_story_file_with_cycles(tmpdir, default_domain):
    graph = await training.extract_story_graph(
        "data/test_stories/stories_with_cycle.md", default_domain
    )

    assert len(graph.story_steps) == 5

    graph_without_cycles = graph.with_cycles_removed()

    assert graph.cyclic_edge_ids != set()
    assert graph_without_cycles.cyclic_edge_ids == set()

    assert len(graph.story_steps) == len(graph_without_cycles.story_steps) == 5

    assert len(graph_without_cycles.story_end_checkpoints) == 2


async def test_generate_training_data_with_cycles(tmpdir, default_domain):
    featurizer = MaxHistoryTrackerFeaturizer(
        BinarySingleStateFeaturizer(), max_history=4
    )
    training_trackers = await training.load_data(
        "data/test_stories/stories_with_cycle.md", default_domain, augmentation_factor=0
    )
    training_data = featurizer.featurize_trackers(training_trackers, default_domain)
    y = training_data.y.argmax(axis=-1)

    # how many there are depends on the graph which is not created in a
    # deterministic way but should always be 3 or 4
    assert len(training_trackers) == 3 or len(training_trackers) == 4

    # if we have 4 trackers, there is going to be one example more for label 4
    num_threes = len(training_trackers) - 1
    # if new default actions are added the keys of the actions will be changed

    assert Counter(y) == {0: 6, 9: 3, 8: num_threes, 1: 2, 10: 1}


async def test_generate_training_data_with_unused_checkpoints(tmpdir, default_domain):
    training_trackers = await training.load_data(
        "data/test_stories/stories_unused_checkpoints.md", default_domain
    )
    # there are 3 training stories:
    #   2 with unused end checkpoints -> training_trackers
    #   1 with unused start checkpoints -> ignored
    assert len(training_trackers) == 2


async def test_generate_training_data_original_and_augmented_trackers(default_domain):
    training_trackers = await training.load_data(
        "data/test_stories/stories_defaultdomain.md",
        default_domain,
        augmentation_factor=3,
    )
    # there are three original stories
    # augmentation factor of 3 indicates max of 3*10 augmented stories generated
    # maximum number of stories should be augmented+original = 33
    original_trackers = [
        t
        for t in training_trackers
        if not hasattr(t, "is_augmented") or not t.is_augmented
    ]
    assert len(original_trackers) == 3
    assert len(training_trackers) <= 33


async def test_visualize_training_data_graph(tmpdir, default_domain):
    graph = await training.extract_story_graph(
        "data/test_stories/stories_with_cycle.md", default_domain
    )

    graph = graph.with_cycles_removed()

    out_path = tmpdir.join("graph.html").strpath

    # this will be the plotted networkx graph
    G = graph.visualize(out_path)

    assert os.path.exists(out_path)

    # we can't check the exact topology - but this should be enough to ensure
    # the visualisation created a sane graph
    assert set(G.nodes()) == set(range(-1, 13)) or set(G.nodes()) == set(range(-1, 14))
    if set(G.nodes()) == set(range(-1, 13)):
        assert len(G.edges()) == 14
    elif set(G.nodes()) == set(range(-1, 14)):
        assert len(G.edges()) == 16


async def test_load_multi_file_training_data(default_domain):
    # the stories file in `data/test_multifile_stories` is the same as in
    # `data/test_stories/stories.md`, but split across multiple files
    featurizer = MaxHistoryTrackerFeaturizer(
        BinarySingleStateFeaturizer(), max_history=2
    )
    trackers = await training.load_data(
        "data/test_stories/stories.md", default_domain, augmentation_factor=0
    )
    (tr_as_sts, tr_as_acts) = featurizer.training_states_and_actions(
        trackers, default_domain
    )
    hashed = []
    for sts, acts in zip(tr_as_sts, tr_as_acts):
        hashed.append(json.dumps(sts + acts, sort_keys=True))
    hashed = sorted(hashed, reverse=True)

    data = featurizer.featurize_trackers(trackers, default_domain)

    featurizer_mul = MaxHistoryTrackerFeaturizer(
        BinarySingleStateFeaturizer(), max_history=2
    )
    trackers_mul = await training.load_data(
        "data/test_multifile_stories", default_domain, augmentation_factor=0
    )
    (tr_as_sts_mul, tr_as_acts_mul) = featurizer.training_states_and_actions(
        trackers_mul, default_domain
    )
    hashed_mul = []
    for sts_mul, acts_mul in zip(tr_as_sts_mul, tr_as_acts_mul):
        hashed_mul.append(json.dumps(sts_mul + acts_mul, sort_keys=True))
    hashed_mul = sorted(hashed_mul, reverse=True)

    data_mul = featurizer_mul.featurize_trackers(trackers_mul, default_domain)

    assert hashed == hashed_mul

    assert np.all(data.X.sort(axis=0) == data_mul.X.sort(axis=0))
    assert np.all(data.y.sort(axis=0) == data_mul.y.sort(axis=0))


async def test_load_training_data_handles_hidden_files(tmpdir, default_domain):
    # create a hidden file

    with open(os.path.join(tmpdir.strpath, ".hidden"), "a") as f:
        f.close()
    # create a normal file
    normal_file = os.path.join(tmpdir.strpath, "normal_file")
    with open(normal_file, "a") as f:
        f.close()

    featurizer = MaxHistoryTrackerFeaturizer(
        BinarySingleStateFeaturizer(), max_history=2
    )
    trackers = await training.load_data(tmpdir.strpath, default_domain)
    data = featurizer.featurize_trackers(trackers, default_domain)

    assert len(data.X) == 0
    assert len(data.y) == 0
