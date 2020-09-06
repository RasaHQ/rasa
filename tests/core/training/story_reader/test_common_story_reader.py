import json
import os
from collections import Counter
from pathlib import Path
from typing import Text, List

import numpy as np
import pytest

from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.events import UserUttered, ActionExecuted, SessionStarted
from rasa.core.featurizers import (
    MaxHistoryTrackerFeaturizer,
    BinarySingleStateFeaturizer,
)
from rasa.nlu.constants import INTENT_NAME_KEY


@pytest.mark.parametrize(
    "stories_file",
    ["data/test_stories/stories.md", "data/test_yaml_stories/stories.yml"],
)
async def test_can_read_test_story(stories_file: Text, default_domain: Domain):
    trackers = await training.load_data(
        stories_file,
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
        intent={INTENT_NAME_KEY: "simple", "confidence": 1.0},
        parse_data={
            "text": "/simple",
            "intent_ranking": [{"confidence": 1.0, INTENT_NAME_KEY: "simple"}],
            "intent": {"confidence": 1.0, INTENT_NAME_KEY: "simple"},
            "entities": [],
        },
    )
    assert tracker.events[2] == ActionExecuted("utter_default")
    assert tracker.events[3] == ActionExecuted("utter_greet")
    assert tracker.events[4] == ActionExecuted("action_listen")


@pytest.mark.parametrize(
    "stories_file",
    [
        "data/test_stories/stories_checkpoint_after_or.md",
        "data/test_yaml_stories/stories_checkpoint_after_or.yml",
    ],
)
async def test_can_read_test_story_with_checkpoint_after_or(
    stories_file: Text, default_domain: Domain
):
    trackers = await training.load_data(
        stories_file,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 2


@pytest.mark.parametrize(
    "stories_file",
    [
        "data/test_stories/stories_with_cycle.md",
        "data/test_yaml_stories/stories_with_cycle.yml",
    ],
)
async def test_read_story_file_with_cycles(stories_file: Text, default_domain: Domain):
    graph = await training.extract_story_graph(stories_file, default_domain)

    assert len(graph.story_steps) == 5

    graph_without_cycles = graph.with_cycles_removed()

    assert graph.cyclic_edge_ids != set()
    # sorting removed_edges converting set converting it to list
    assert graph_without_cycles.cyclic_edge_ids == list()

    assert len(graph.story_steps) == len(graph_without_cycles.story_steps) == 5

    assert len(graph_without_cycles.story_end_checkpoints) == 2


@pytest.mark.parametrize(
    "stories_file",
    [
        "data/test_stories/stories_with_cycle.md",
        "data/test_yaml_stories/stories_with_cycle.yml",
    ],
)
async def test_generate_training_data_with_cycles(
    stories_file: Text, default_domain: Domain
):
    featurizer = MaxHistoryTrackerFeaturizer(
        BinarySingleStateFeaturizer(), max_history=4
    )
    training_trackers = await training.load_data(
        stories_file, default_domain, augmentation_factor=0
    )
    training_data = featurizer.featurize_trackers(training_trackers, default_domain)
    y = training_data.y.argmax(axis=-1)

    # how many there are depends on the graph which is not created in a
    # deterministic way but should always be 3 or 4
    assert len(training_trackers) == 3 or len(training_trackers) == 4

    # if we have 4 trackers, there is going to be one example more for label 10
    num_tens = len(training_trackers) - 1
    # if new default actions are added the keys of the actions will be changed

    assert Counter(y) == {0: 6, 12: num_tens, 14: 1, 1: 2, 13: 3}


@pytest.mark.parametrize(
    "stories_file",
    [
        "data/test_stories/stories_unused_checkpoints.md",
        "data/test_yaml_stories/stories_unused_checkpoints.yml",
    ],
)
async def test_generate_training_data_with_unused_checkpoints(
    stories_file: Text, default_domain: Domain
):
    training_trackers = await training.load_data(stories_file, default_domain)
    # there are 3 training stories:
    #   2 with unused end checkpoints -> training_trackers
    #   1 with unused start checkpoints -> ignored
    assert len(training_trackers) == 2


@pytest.mark.parametrize(
    "stories_file",
    [
        "data/test_stories/stories_defaultdomain.md",
        "data/test_yaml_stories/stories_defaultdomain.yml",
    ],
)
async def test_generate_training_data_original_and_augmented_trackers(
    stories_file: Text, default_domain: Domain
):
    training_trackers = await training.load_data(
        stories_file, default_domain, augmentation_factor=3
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


@pytest.mark.parametrize(
    "stories_file",
    [
        "data/test_stories/stories_with_cycle.md",
        "data/test_yaml_stories/stories_with_cycle.yml",
    ],
)
async def test_visualize_training_data_graph(
    stories_file: Text, tmp_path: Path, default_domain: Domain
):
    graph = await training.extract_story_graph(stories_file, default_domain)

    graph = graph.with_cycles_removed()

    out_path = str(tmp_path / "graph.html")

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


@pytest.mark.parametrize(
    "stories_resources",
    [
        ["data/test_stories/stories.md", "data/test_multifile_stories"],
        ["data/test_yaml_stories/stories.yml", "data/test_multifile_yaml_stories"],
        ["data/test_stories/stories.md", "data/test_multifile_yaml_stories"],
        ["data/test_yaml_stories/stories.yml", "data/test_multifile_stories"],
        ["data/test_stories/stories.md", "data/test_mixed_yaml_md_stories"],
    ],
)
async def test_load_multi_file_training_data(
    stories_resources: List, default_domain: Domain
):
    # the stories file in `data/test_multifile_stories` is the same as in
    # `data/test_stories/stories.md`, but split across multiple files
    featurizer = MaxHistoryTrackerFeaturizer(
        BinarySingleStateFeaturizer(), max_history=2
    )
    trackers = await training.load_data(
        stories_resources[0], default_domain, augmentation_factor=0
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
        stories_resources[1], default_domain, augmentation_factor=0
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


async def test_load_training_data_reader_not_found_throws(
    tmp_path: Path, default_domain: Domain
):
    (tmp_path / "file").touch()

    with pytest.raises(Exception):
        await training.load_data(str(tmp_path), default_domain)


def test_session_started_event_is_not_serialised():
    assert SessionStarted().as_story_string() is None
