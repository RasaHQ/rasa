import os

import json
from collections import Counter
from pathlib import Path
from typing import Text, Dict

import numpy as np
import pytest

import rasa.utils.io
from rasa.core import training, utils
from rasa.core.interpreter import RegexInterpreter
from rasa.core.training.dsl import StoryFileReader, EndToEndReader
from rasa.core.domain import Domain
from rasa.core.trackers import DialogueStateTracker
from rasa.core.events import (
    UserUttered,
    ActionExecuted,
    ActionExecutionRejected,
    Form,
    FormValidation,
    SessionStarted,
)
from rasa.core.training.structures import Story
from rasa.core.featurizers import (
    MaxHistoryTrackerFeaturizer,
    BinarySingleStateFeaturizer,
)
from rasa.utils.io import DEFAULT_ENCODING


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
    rasa.utils.io.write_text_file(graph.as_story_string(), out_path.strpath)

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


async def test_persist_form_story(tmpdir):
    domain = Domain.load("data/test_domains/form.yml")

    tracker = DialogueStateTracker("", domain.slots)

    story = (
        "* greet\n"
        "    - utter_greet\n"
        "* start_form\n"
        "    - some_form\n"
        '    - form{"name": "some_form"}\n'
        "* default\n"
        "    - utter_default\n"
        "    - some_form\n"
        "* stop\n"
        "    - utter_ask_continue\n"
        "* affirm\n"
        "    - some_form\n"
        "* stop\n"
        "    - utter_ask_continue\n"
        "    - action_listen\n"
        "* form: inform\n"
        "    - some_form\n"
        '    - form{"name": null}\n'
        "* goodbye\n"
        "    - utter_goodbye\n"
    )

    # simulate talking to the form
    events = [
        UserUttered(intent={"name": "greet"}),
        ActionExecuted("utter_greet"),
        ActionExecuted("action_listen"),
        # start the form
        UserUttered(intent={"name": "start_form"}),
        ActionExecuted("some_form"),
        Form("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "default"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_default"),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "stop"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_ask_continue"),
        ActionExecuted("action_listen"),
        # out of form input but continue with the form
        UserUttered(intent={"name": "affirm"}),
        FormValidation(False),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "stop"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_ask_continue"),
        ActionExecuted("action_listen"),
        # form input
        UserUttered(intent={"name": "inform"}),
        FormValidation(True),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        Form(None),
        UserUttered(intent={"name": "goodbye"}),
        ActionExecuted("utter_goodbye"),
        ActionExecuted("action_listen"),
    ]
    [tracker.update(e) for e in events]

    assert story in tracker.export_stories()


async def test_read_story_file_with_cycles(tmpdir, default_domain):
    graph = await training.extract_story_graph(
        "data/test_stories/stories_with_cycle.md", default_domain
    )

    assert len(graph.story_steps) == 5

    graph_without_cycles = graph.with_cycles_removed()

    assert graph.cyclic_edge_ids != set()
    # sorting removed_edges converting set converting it to list
    assert graph_without_cycles.cyclic_edge_ids == list()

    assert len(graph.story_steps) == len(graph_without_cycles.story_steps) == 5

    assert len(graph_without_cycles.story_end_checkpoints) == 2


async def test_generate_training_data_with_cycles(default_domain):
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

    # if we have 4 trackers, there is going to be one example more for label 10
    num_tens = len(training_trackers) - 1
    # if new default actions are added the keys of the actions will be changed

    assert Counter(y) == {0: 6, 10: num_tens, 12: 1, 1: 2, 11: 3}


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
    Path(tmpdir / ".hidden").touch()
    # create a normal file
    Path(tmpdir / "normal_file").touch()

    featurizer = MaxHistoryTrackerFeaturizer(
        BinarySingleStateFeaturizer(), max_history=2
    )
    trackers = await training.load_data(tmpdir.strpath, default_domain)
    data = featurizer.featurize_trackers(trackers, default_domain)

    assert len(data.X) == 0
    assert len(data.y) == 0


async def test_read_stories_with_multiline_comments(tmpdir, default_domain):
    story_steps = await StoryFileReader.read_from_file(
        "data/test_stories/stories_with_multiline_comments.md",
        default_domain,
        RegexInterpreter(),
    )

    assert len(story_steps) == 4
    assert story_steps[0].block_name == "happy path"
    assert len(story_steps[0].events) == 4
    assert story_steps[1].block_name == "sad path 1"
    assert len(story_steps[1].events) == 7
    assert story_steps[2].block_name == "sad path 2"
    assert len(story_steps[2].events) == 7
    assert story_steps[3].block_name == "say goodbye"
    assert len(story_steps[3].events) == 2


@pytest.mark.parametrize(
    "line, expected",
    [
        (" greet: hi", {"intent": "greet", "true_intent": "greet", "text": "hi"}),
        (
            " greet: /greet",
            {
                "intent": "greet",
                "true_intent": "greet",
                "text": "/greet",
                "entities": [],
            },
        ),
        (
            'greet: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"entity": "test", "start": 6, "end": 22, "value": "test"}
                ],
                "true_intent": "greet",
                "text": '/greet{"test": "test"}',
            },
        ),
        (
            'greet{"test": "test"}: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"entity": "test", "start": 6, "end": 22, "value": "test"}
                ],
                "true_intent": "greet",
                "text": '/greet{"test": "test"}',
            },
        ),
        (
            "mood_great: [great](feeling)",
            {
                "intent": "mood_great",
                "entities": [
                    {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                ],
                "true_intent": "mood_great",
                "text": "great",
            },
        ),
        (
            'form: greet{"test": "test"}: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"end": 22, "entity": "test", "start": 6, "value": "test"}
                ],
                "true_intent": "greet",
                "text": '/greet{"test": "test"}',
            },
        ),
    ],
)
def test_e2e_parsing(line: Text, expected: Dict):
    reader = EndToEndReader()
    actual = reader._parse_item(line)

    assert actual.as_dict() == expected


@pytest.mark.parametrize(
    "parse_data, expected_story_string",
    [
        (
            {
                "text": "/simple",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [
                        {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                    ],
                },
            },
            "simple: /simple",
        ),
        (
            {
                "text": "great",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [
                        {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                    ],
                },
            },
            "simple: [great](feeling)",
        ),
        (
            {
                "text": "great",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [],
                },
            },
            "simple: great",
        ),
    ],
)
def test_user_uttered_to_e2e(parse_data: Dict, expected_story_string: Text):
    event = UserUttered.from_story_string("user", parse_data)[0]

    assert isinstance(event, UserUttered)
    assert event.as_story_string(e2e=True) == expected_story_string


def test_session_started_event_is_not_serialised():
    assert SessionStarted().as_story_string() is None


@pytest.mark.parametrize("line", [" greet{: hi"])
def test_invalid_end_to_end_format(line: Text):
    reader = EndToEndReader()

    with pytest.raises(ValueError):
        # noinspection PyProtectedMember
        _ = reader._parse_item(line)
