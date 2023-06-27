import csv
from typing import List, Optional, Set, Text, Tuple, Type, Any
import itertools
from pathlib import Path

import pytest
import numpy as np

import rasa.shared.utils.io
from rasa.core.evaluation.marker import (
    IntentDetectedMarker,
    SlotSetMarker,
    OccurrenceMarker,
    ActionExecutedMarker,
    AndMarker,
    OrMarker,
    NotMarker,
    SequenceMarker,
)
from rasa.core.evaluation.marker_base import (
    OperatorMarker,
    Marker,
    ConditionMarker,
    InvalidMarkerConfig,
)
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.events import SlotSet, ActionExecuted, UserUttered, SessionStarted
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.shared.core.slots import TextSlot
from rasa.core.evaluation.marker_tracker_loader import MarkerTrackerLoader
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.domain import Domain

CONDITION_MARKERS = [ActionExecutedMarker, SlotSetMarker, IntentDetectedMarker]
OPERATOR_MARKERS = [AndMarker, OrMarker, SequenceMarker, OccurrenceMarker]


def test_marker_from_config():
    config = {
        AndMarker.positive_tag(): [
            {SlotSetMarker.positive_tag(): "s1"},
            {
                OrMarker.positive_tag(): [
                    {IntentDetectedMarker.positive_tag(): "4"},
                    {IntentDetectedMarker.negated_tag(): "6"},
                ]
            },
        ]
    }

    marker = Marker.from_config(config)
    assert isinstance(marker, AndMarker)
    assert isinstance(marker.sub_markers[0], SlotSetMarker)
    or_marker = marker.sub_markers[1]
    assert isinstance(or_marker, OrMarker)
    for sub_marker in or_marker.sub_markers:
        assert isinstance(sub_marker, ConditionMarker)


@pytest.mark.parametrize("marker_class", CONDITION_MARKERS)
def test_condition_negated_to_str(marker_class: Type[ConditionMarker]):
    marker = marker_class("intent1", negated=True)
    if marker.negated_tag() is not None:
        assert marker.negated_tag() in str(marker)


@pytest.mark.parametrize(
    "condition_marker_type, negated",
    itertools.product(CONDITION_MARKERS, [False, True]),
)
def test_condition(condition_marker_type: Type[ConditionMarker], negated: bool):
    """Each marker applies an exact number of times (slots are immediately un-set)."""
    marker = condition_marker_type(
        text="same-text", name="marker_name", negated=negated
    )
    events = [
        UserUttered(intent={"name": "1"}),
        UserUttered(intent={"name": "same-text"}),
        SlotSet("same-text", value="any"),
        SlotSet("same-text", value=None),
        ActionExecuted(action_name="same-text"),
    ]
    num_non_negated_condition_applies = 3
    events = events * num_non_negated_condition_applies
    for event in events:
        marker.track(event)
    assert len(marker.history) == len(events)
    expected = (
        num_non_negated_condition_applies
        if not negated
        else (len(events) - num_non_negated_condition_applies)
    )
    assert sum(marker.history) == expected


@pytest.mark.parametrize("condition_marker_type", CONDITION_MARKERS)
def test_condition_evaluate_events(condition_marker_type: Type[ConditionMarker]):
    """Each marker applies an exact number of times (slots are immediately un-set)."""
    events = [
        UserUttered(intent={INTENT_NAME_KEY: "1"}),
        UserUttered(intent={INTENT_NAME_KEY: "same-text"}),
        SlotSet("same-text", value="any"),
        SlotSet("same-text", value=None),
        ActionExecuted(action_name="same-text"),
    ]
    num_applies = 3
    events = events * num_applies
    marker = condition_marker_type(text="same-text", name="marker_name")
    evaluation = marker.evaluate_events(events)
    assert len(evaluation) == 1
    assert "marker_name" in evaluation[0]
    if condition_marker_type == IntentDetectedMarker:
        expected = [1, 3, 5]
    else:
        expected = [2, 4, 6]

    actual_preceding_user_turns = [
        meta_data.preceding_user_turns for meta_data in evaluation[0]["marker_name"]
    ]
    assert actual_preceding_user_turns == expected


@pytest.mark.parametrize("marker_class", OPERATOR_MARKERS)
def test_operator_negated_to_str(marker_class: Type[OperatorMarker]):
    marker = marker_class([IntentDetectedMarker("bla")], negated=True)
    if marker.negated_tag() is not None:
        assert marker.negated_tag() in str(marker)


@pytest.mark.parametrize(
    "operator_class, negated",
    [
        (operator_class, negated)
        for operator_class, negated in itertools.product(
            OPERATOR_MARKERS, [True, False]
        )
        if operator_class.expected_number_of_sub_markers() is not None
    ],
)
def test_operator_raises_wrong_amount_sub_markers(
    operator_class: Type[OperatorMarker], negated: bool
):
    expected_number = operator_class.expected_number_of_sub_markers()
    one_more_than_expected = [
        IntentDetectedMarker("bla") for _ in range(expected_number + 1)
    ]
    with pytest.raises(InvalidMarkerConfig):
        operator_class(one_more_than_expected, name="marker_name", negated=negated)


@pytest.mark.parametrize("negated", [True, False])
def test_operator_or(negated: bool):
    events = [
        UserUttered(intent={INTENT_NAME_KEY: "1"}),
        UserUttered(intent={INTENT_NAME_KEY: "unknown"}),
        UserUttered(intent={INTENT_NAME_KEY: "2"}),
        UserUttered(intent={INTENT_NAME_KEY: "unknown"}),
    ]
    sub_markers = [IntentDetectedMarker("1"), IntentDetectedMarker("2")]
    marker = OrMarker(sub_markers, name="marker_name", negated=negated)
    for event in events:
        marker.track(event)
    expected = [True, False, True, False]
    if negated:
        expected = [not applies for applies in expected]
    assert marker.history == expected


@pytest.mark.parametrize("negated", [True, False])
def test_operator_not(negated: bool):
    events = [
        UserUttered(intent={INTENT_NAME_KEY: "1"}),
        UserUttered(intent={INTENT_NAME_KEY: "unknown"}),
    ]
    sub_markers = [IntentDetectedMarker("1")]
    marker = NotMarker(sub_markers, name="marker_name", negated=negated)
    for event in events:
        marker.track(event)
    expected = [False, True]
    if negated:
        expected = [not applies for applies in expected]
    assert marker.history == expected


@pytest.mark.parametrize("negated", [True, False])
def test_operator_and(negated: bool):
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (SlotSet("2", value="bla"), False),
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), True),
        (SlotSet("2", value=None), False),
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (SlotSet("2", value="bla"), False),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), False),
    ]
    events, expected = zip(*events_expected)
    sub_markers = [IntentDetectedMarker("1"), SlotSetMarker("2")]
    marker = AndMarker(sub_markers, name="marker_name", negated=negated)
    for event in events:
        marker.track(event)
    expected = list(expected)
    if negated:
        expected = [not applies for applies in expected]
    assert marker.history == expected


@pytest.mark.parametrize("negated", [False, True])
def test_operator_seq(negated: bool):
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (ActionExecuted("unrelated event that does not interrupt the sequence"), False),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), True),
        (UserUttered(intent={INTENT_NAME_KEY: "3"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), True),
    ]
    events, expected = zip(*events_expected)
    sub_markers = [IntentDetectedMarker("1"), IntentDetectedMarker("2")]
    marker = SequenceMarker(sub_markers, name="marker_name", negated=negated)
    for event in events:
        marker.track(event)
    expected = list(expected)
    if negated:
        expected = [not applies for applies in expected]
    assert marker.history == expected


@pytest.mark.parametrize("negated", [False, True])
def test_operator_seq_does_not_allow_overlap(negated: bool):
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "3"}), True),
        (UserUttered(intent={INTENT_NAME_KEY: "3"}), False),
    ]
    events, expected = zip(*events_expected)
    sub_markers = [
        IntentDetectedMarker("1"),
        IntentDetectedMarker("2"),
        IntentDetectedMarker("3"),
    ]
    marker = SequenceMarker(sub_markers, name="marker_name", negated=negated)
    for event in events:
        marker.track(event)
    expected = list(expected)
    if negated:
        expected = [not applies for applies in expected]
    assert marker.history == expected


@pytest.mark.parametrize("negated", [True, False])
def test_operator_occur(negated: bool):
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "0"}), False),
        (SlotSet("2", value=None), False),
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), True),
        (SlotSet("2", value=None), True),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), True),
        (SlotSet("2", value="bla"), True),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), True),
    ]
    events, expected = zip(*events_expected)
    sub_marker = OrMarker(
        [IntentDetectedMarker("1"), SlotSetMarker("2")], name="or marker", negated=False
    )
    marker = OccurrenceMarker([sub_marker], name="marker_name", negated=negated)
    for event in events:
        marker.track(event)
    expected = list(expected)
    if negated:
        expected = [not applies for applies in expected]
    assert marker.history == expected

    assert marker.relevant_events() == [expected.index(True)]


def test_operator_occur_never_applied():
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), False),
        (SlotSet("2", value=None), False),
        (UserUttered(intent={INTENT_NAME_KEY: "0"}), False),
        (SlotSet("1", value="test"), True),
    ]
    events, expected = zip(*events_expected)
    sub_marker = OrMarker(
        [IntentDetectedMarker("1"), SlotSetMarker("2")],
        name="and marker",
        negated=False,
    )
    marker = OccurrenceMarker([sub_marker], name="or_at_some_point", negated=False)
    for event in events:
        marker.track(event)

    assert marker.relevant_events() == []


def test_operator_occur_never_applied_negated():
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (SlotSet("2", value=None), False),
        (UserUttered(intent={INTENT_NAME_KEY: "0"}), False),
        (SlotSet("1", value="test"), False),
    ]
    events, expected = zip(*events_expected)
    sub_marker = OrMarker(
        [IntentDetectedMarker("1"), SlotSetMarker("2")], name="or marker", negated=False
    )
    marker = OccurrenceMarker([sub_marker], name="or never occurred", negated=True)
    for event in events:
        marker.track(event)

    assert marker.relevant_events() == []


def test_operators_nested_simple():
    events = [
        UserUttered(intent={"name": "1"}),
        UserUttered(intent={"name": "2"}),
        UserUttered(intent={"name": "3"}),
        SlotSet("s1", value="any"),
        UserUttered(intent={"name": "4"}),
        UserUttered(intent={"name": "5"}),
        UserUttered(intent={"name": "6"}),
    ]
    marker = AndMarker(
        markers=[
            SlotSetMarker("s1"),
            OrMarker([IntentDetectedMarker("4"), IntentDetectedMarker("6")]),
        ],
        name="marker_name",
    )
    evaluation = marker.evaluate_events(events)

    assert len(evaluation[0]["marker_name"]) == 2
    assert evaluation[0]["marker_name"][0].preceding_user_turns == 3
    assert evaluation[0]["marker_name"][1].preceding_user_turns == 5


def generate_random_marker(
    depth: int,
    max_branches: int,
    rng: np.random.Generator,
    constant_condition_text: Optional[Text],
    possible_conditions: List[Type[ConditionMarker]],
    constant_negated: Optional[bool],
    possible_operators: List[Type[OperatorMarker]],
) -> Tuple[Marker, int]:
    """Generates an (max_branches)-ary tree with the specified depth."""
    if depth == 0:
        condition_class = possible_conditions[rng.choice(len(possible_conditions))]
        negated = bool(rng.choice(2)) if constant_negated is None else constant_negated
        condition_text = constant_condition_text or f"{rng.choice(1000)}"
        return condition_class(text=condition_text, negated=negated), 1
    else:
        negated = bool(rng.choice(2)) if constant_negated is None else constant_negated
        operator_class = possible_operators[rng.choice(len(possible_operators))]
        num_branches = operator_class.expected_number_of_sub_markers()
        if num_branches is None:
            num_branches = rng.choice(max_branches - 1) + 1
        marker_size = 0
        sub_markers = []
        for _ in range(num_branches):
            sub_marker, sub_marker_size = generate_random_marker(
                depth=depth - 1,
                max_branches=max_branches,
                rng=rng,
                constant_negated=constant_negated,
                constant_condition_text=constant_condition_text,
                possible_operators=possible_operators,
                possible_conditions=possible_conditions,
            )
            marker_size += sub_marker_size
            sub_markers.append(sub_marker)

        marker = operator_class(markers=sub_markers, negated=negated)
        marker_size += 1
        return marker, marker_size


@pytest.mark.parametrize(
    "depth, max_branches, seed", [(1, 3, 3456), (4, 3, 345), (4, 5, 2345)]
)
def test_operator_nested_randomly_all_sub_markers_track_events(
    depth: int, max_branches: int, seed: int
):
    rng = np.random.default_rng(seed=seed)
    marker, expected_size = generate_random_marker(
        depth=depth,
        max_branches=max_branches,
        rng=rng,
        possible_conditions=CONDITION_MARKERS,
        possible_operators=OPERATOR_MARKERS,
        constant_condition_text=None,
        constant_negated=None,
    )
    events = [
        UserUttered(intent={"name": "1"}),
        UserUttered(intent={"name": "1"}),
        SlotSet("1", value="any"),
        SlotSet("1", value=None),
        ActionExecuted(action_name="1"),
    ]
    for event in events:
        marker.track(event)
    assert len([sub_marker for sub_marker in marker.flatten()]) == expected_size
    for sub_marker in marker.flatten():
        assert len(sub_marker.history) == len(events)


@pytest.mark.parametrize(
    "depth, seed, applies_at_some_point",
    [
        (depth, seed, applies_at_some_point)
        for depth, seed in [(1, 3456), (4, 345)]
        for applies_at_some_point in [True, False]
    ],
)
def test_operator_nested_randomly_all_sub_markers_track_events_and_apply_at_some_point(
    depth: int, seed: int, applies_at_some_point: bool
):
    rng = np.random.default_rng(seed=seed)
    constant_condition_text = "1" if applies_at_some_point else "2"
    marker, _ = generate_random_marker(
        depth=depth,
        max_branches=3,
        rng=rng,
        possible_conditions=CONDITION_MARKERS,
        possible_operators=[OrMarker],
        constant_negated=False,
        constant_condition_text=constant_condition_text,
    )
    # By setting the max_branches to 3 and then tracking all permutations of these
    # 3 events we ensure that all markers apply at some point (including `seq`)
    events = [
        UserUttered(intent={"name": "1"}),
        SlotSet("1", value="any"),
        ActionExecuted(action_name="1"),
    ]
    for permuted_events in itertools.permutations(events):
        for event in permuted_events:
            marker.track(event)

    # by design, every marker applies at some point / never
    if applies_at_some_point:
        assert all([any(sub_marker.history) for sub_marker in marker.flatten()])
    else:
        assert all([not any(sub_marker.history) for sub_marker in marker.flatten()])


def test_sessions_evaluated_separately():
    """Each marker applies an exact number of times (slots are immediately un-set)."""

    events = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        UserUttered(intent={INTENT_NAME_KEY: "ignored"}),
        UserUttered(intent={INTENT_NAME_KEY: "ignored"}),
        UserUttered(intent={INTENT_NAME_KEY: "ignored"}),
        SlotSet("same-text", value="any"),
        ActionExecuted(action_name=ACTION_SESSION_START_NAME),
        UserUttered(intent={INTENT_NAME_KEY: "no-slot-set-here"}),
        UserUttered(intent={INTENT_NAME_KEY: "no-slot-set-here"}),
    ]

    marker = SlotSetMarker(text="same-text", name="my-marker")
    evaluation = marker.evaluate_events(events)

    assert len(evaluation) == 2
    assert len(evaluation[0]["my-marker"]) == 1
    assert evaluation[0]["my-marker"][0].preceding_user_turns == 3
    assert len(evaluation[1]["my-marker"]) == 0  # i.e. slot set does not "leak"


def test_sessions_evaluated_returns_event_indices_wrt_tracker_not_dialogue():
    events = [
        ActionExecuted(action_name=ACTION_SESSION_START_NAME),
        UserUttered(intent={INTENT_NAME_KEY: "ignored"}),
        UserUttered(intent={INTENT_NAME_KEY: "ignored"}),
        UserUttered(intent={INTENT_NAME_KEY: "ignored"}),
        SlotSet("same-text", value="any"),
        ActionExecuted(action_name=ACTION_SESSION_START_NAME),
        UserUttered(intent={INTENT_NAME_KEY: "no-slot-set-here"}),
        UserUttered(intent={INTENT_NAME_KEY: "no-slot-set-here"}),
        SlotSet("same-text", value="any"),
    ]
    marker = SlotSetMarker(text="same-text", name="my-marker")
    evaluation = marker.evaluate_events(events)
    assert len(evaluation) == 2
    assert len(evaluation[0]["my-marker"]) == 1
    assert evaluation[0]["my-marker"][0].preceding_user_turns == 3
    assert evaluation[0]["my-marker"][0].idx == 4
    assert len(evaluation[1]["my-marker"]) == 1
    assert evaluation[1]["my-marker"][0].preceding_user_turns == 2
    assert evaluation[1]["my-marker"][0].idx == 8  # i.e. NOT the index in the dialogue


async def test_markers_cli_results_save_correctly(tmp_path: Path):
    domain = Domain.empty()
    store = InMemoryTrackerStore(domain)

    for i in range(5):
        tracker = DialogueStateTracker(str(i), None)
        tracker.update_with_events([SlotSet(str(j), "slot") for j in range(5)], domain)
        tracker.update(ActionExecuted(ACTION_SESSION_START_NAME))
        tracker.update(UserUttered("hello"))
        tracker.update_with_events(
            [SlotSet(str(5 + j), "slot") for j in range(5)], domain
        )
        await store.save(tracker)

    tracker_loader = MarkerTrackerLoader(store, "all")

    results_path = tmp_path / "results.csv"

    markers = OrMarker(
        markers=[SlotSetMarker("2", name="marker1"), SlotSetMarker("7", name="marker2")]
    )
    await markers.evaluate_trackers(tracker_loader.load(), results_path)

    with open(results_path, "r") as results:
        result_reader = csv.DictReader(results)
        senders = set()

        for row in result_reader:
            senders.add(row["sender_id"])
            if row["marker"] == "marker1":
                assert row["session_idx"] == "0"
                assert int(row["event_idx"]) >= 2
                assert row["num_preceding_user_turns"] == "0"

            if row["marker"] == "marker2":
                assert row["session_idx"] == "1"
                assert int(row["event_idx"]) >= 3
                assert row["num_preceding_user_turns"] == "1"

        assert len(senders) == 5


def _collect_parameters(
    marker: Marker, condition_type: Type[ConditionMarker]
) -> Set[Text]:
    return set(
        sub_marker.text
        for sub_marker in marker.flatten()
        if isinstance(sub_marker, condition_type)
    )


@pytest.mark.parametrize(
    "depth, max_branches, seed", [(1, 3, 3456), (4, 3, 345), (4, 5, 2345)]
)
def test_domain_validation_with_valid_marker(depth: int, max_branches: int, seed: int):
    # We do this a bit backwards, we construct the domain from the marker
    # and assert they must match
    rng = np.random.default_rng(seed=seed)
    marker, expected_size = generate_random_marker(
        depth=depth,
        max_branches=max_branches,
        rng=rng,
        possible_conditions=CONDITION_MARKERS,
        possible_operators=OPERATOR_MARKERS,
        constant_condition_text=None,
        constant_negated=None,
    )

    slots = [TextSlot(name, []) for name in _collect_parameters(marker, SlotSetMarker)]
    actions = list(_collect_parameters(marker, ActionExecutedMarker))
    intents = _collect_parameters(marker, IntentDetectedMarker)
    domain = Domain(intents, [], slots, {}, actions, {}, {})

    assert marker.validate_against_domain(domain)


@pytest.mark.parametrize(
    "depth, max_branches, seed", [(1, 3, 3456), (4, 3, 345), (4, 5, 2345)]
)
def test_domain_validation_with_invalid_marker(
    depth: int, max_branches: int, seed: int
):
    rng = np.random.default_rng(seed=seed)
    marker, expected_size = generate_random_marker(
        depth=depth,
        max_branches=max_branches,
        rng=rng,
        possible_conditions=CONDITION_MARKERS,
        possible_operators=OPERATOR_MARKERS,
        constant_condition_text=None,
        constant_negated=None,
    )

    domain = Domain.empty()
    with pytest.warns(None):
        is_valid = marker.validate_against_domain(domain)
    assert not is_valid


@pytest.mark.parametrize(
    "config",
    [
        # (1) top level configuration
        # - config is list
        [SlotSetMarker.positive_tag()],
        # - config is list with nested dict
        [{SlotSetMarker.positive_tag(): "s1"}],
        # - config is just a str
        # (2) sub-markers
        SlotSetMarker.positive_tag(),
        # - sub-marker config is dict not list
        {AndMarker.positive_tag(): {SlotSetMarker.positive_tag(): "s1"}},
        # - sub-marker under condition
        {SlotSetMarker.positive_tag(): {IntentDetectedMarker.positive_tag(): "blah"}},
        # - no sub-config for operator (see also: "amount_sub_markers" tests)
        {AndMarker.positive_tag(): "blah"},
        # (3) tags
        # - unknown operator
        {"Ard": {IntentDetectedMarker.positive_tag(): "intent1"}},
        # - unknown condition
        {AndMarker.positive_tag(): {"intent": "intent1"}},
        # -  special tag used by user
        {Marker.ANY_MARKER: "blah"},
    ],
)
def test_marker_validation_raises(config: Any):
    with pytest.raises(InvalidMarkerConfig):
        Marker.from_config(config)


def test_marker_from_path_only_reads_yamls(tmp_path: Path):
    suffixes = [("yaml", True), ("yml", True), ("yaeml", False), ("config", False)]
    for idx, (suffix, allowed) in enumerate(suffixes):
        config = {f"marker-{idx}": {IntentDetectedMarker.positive_tag(): "intent"}}
        config_file = tmp_path / f"config-{idx}.{suffix}"
        rasa.shared.utils.io.write_yaml(data=config, target=config_file)
    loaded = Marker.from_path(tmp_path)
    assert len(loaded.sub_markers) == sum(allowed for _, allowed in suffixes)
    assert set(sub_marker.name for sub_marker in loaded.sub_markers) == set(
        f"marker-{idx}" for idx, (_, allowed) in enumerate(suffixes) if allowed
    )


@pytest.mark.parametrize(
    "configs",
    [
        {"my_marker1": {IntentDetectedMarker.positive_tag(): "this-intent"}},
        {
            "my_marker1": {IntentDetectedMarker.positive_tag(): "this-intent"},
            "my_marker2": {IntentDetectedMarker.positive_tag(): "this-action"},
        },
    ],
)
def test_marker_from_path_adds_special_or_marker(tmp_path: Path, configs: Any):

    yaml_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(data=configs, target=yaml_file)
    loaded = Marker.from_path(tmp_path)
    assert isinstance(loaded, OrMarker)
    assert loaded.name == Marker.ANY_MARKER
    assert len(loaded.sub_markers) == len(configs)
    assert all(
        isinstance(sub_marker, IntentDetectedMarker)
        for sub_marker in loaded.sub_markers
    )


@pytest.mark.parametrize(
    "path_config_tuples",
    [
        [  # two configs defining the same marker
            (
                "folder1/config1.yaml",
                {"my_marker1": {IntentDetectedMarker.positive_tag(): "this-intent"}},
            ),
            (
                "folder2/config2.yaml",
                {
                    "my_marker1": {
                        IntentDetectedMarker.positive_tag(): "different-intent"
                    },
                    "my_marker2": {ActionExecutedMarker.positive_tag(): "action"},
                },
            ),
        ],
        [  # one buggy config
            (
                "folder1/config2.yaml",
                {"my_marker1": {IntentDetectedMarker.positive_tag(): "intent1"}},
            ),
            ("folder2/config2.yaml", {"my_marker2": {"unknown-tag": "intent2"}}),
        ],
    ],
)
def test_marker_from_path_raises(
    tmp_path: Path, path_config_tuples: List[Tuple[Text, Any]]
):
    for path_to_yaml, config in path_config_tuples:
        full_path = tmp_path / path_to_yaml
        folder = full_path.parents[0]
        if folder != tmp_path:
            Path.mkdir(folder, exist_ok=False)
        rasa.shared.utils.io.write_yaml(data=config, target=full_path)
    with pytest.raises(InvalidMarkerConfig):
        Marker.from_path(tmp_path)


@pytest.mark.parametrize(
    "marker,expected_depth",
    [
        (
            AndMarker(
                markers=[
                    SlotSetMarker("s1"),
                    OrMarker([IntentDetectedMarker("4"), IntentDetectedMarker("6")]),
                ]
            ),
            3,
        ),
        (SlotSetMarker("s1"), 1),
        (AndMarker(markers=[SlotSetMarker("s1"), IntentDetectedMarker("6")]), 2),
    ],
)
def test_marker_depth(marker: Marker, expected_depth: int):
    assert marker.max_depth() == expected_depth


def test_split_sessions(tmp_path):
    """Tests loading a tracker with multiple sessions."""

    events = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        UserUttered(intent={"name": "this-intent"}),
    ]
    sessions = Marker._split_sessions(events)
    assert len(sessions) == 1
    assert len(sessions[0][0]) == len(events)


def test_condition_marker_with_description():
    marker = SlotSetMarker("s1", description="This is a description")
    assert marker.description == "This is a description"


def test_operator_marker_with_description():
    marker = AndMarker(
        markers=[SlotSetMarker("s1")], description="This is a description"
    )
    assert marker.description == "This is a description"
