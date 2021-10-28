from typing import List, Optional, Text, Tuple, Type
import itertools

import pytest
import numpy as np

import rasa.shared.utils.schemas.markers
from rasa.core.evaluation.marker import (
    ActionExecutedMarker,
    AndMarker,
    IntentDetectedMarker,
    OrMarker,
    SlotSetMarker,
    SequenceMarker,
    OccurrenceMarker,
)
from rasa.core.evaluation.marker_base import (
    CompoundMarker,
    Marker,
    AtomicMarker,
)
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.events import SlotSet, ActionExecuted, UserUttered
from rasa.shared.nlu.constants import INTENT_NAME_KEY


CONDITION_MARKERS = [ActionExecutedMarker, SlotSetMarker, IntentDetectedMarker]
OPERATOR_MARKERS = [AndMarker, OrMarker, SequenceMarker, OccurrenceMarker]


def test_marker_from_config_dict_single_and():
    config = {
        "marker_1": {
            AndMarker.tag(): [
                {SlotSetMarker.tag(): ["s1"]},
                {
                    OrMarker.tag(): [
                        {IntentDetectedMarker.tag(): ["4"]},
                        {IntentDetectedMarker.negated_tag(): ["6"]},
                    ]
                },
            ]
        }
    }
    marker = Marker.from_config_dict(config)
    assert marker.name == "marker_1"
    assert isinstance(marker, AndMarker)
    assert isinstance(marker.sub_markers[0], SlotSetMarker)
    assert isinstance(marker.sub_markers[1], OrMarker)
    for sub_marker in marker.sub_markers[1].sub_markers:
        assert isinstance(sub_marker, AtomicMarker)


def test_marker_from_config_list_inserts_and_marker():
    config = [
        {SlotSetMarker.tag(): ["s1"]},
        {
            OrMarker.tag(): [
                {IntentDetectedMarker.tag(): ["4"]},
                {IntentDetectedMarker.negated_tag(): ["6"]},
            ]
        },
    ]
    marker = Marker.from_config(config)
    assert isinstance(marker, AndMarker)  # i.e. the default marker inserted
    assert isinstance(marker.sub_markers[0], SlotSetMarker)
    assert isinstance(marker.sub_markers[1], OrMarker)
    for sub_marker in marker.sub_markers[1].sub_markers:
        assert isinstance(sub_marker, AtomicMarker)


def test_marker_from_config_unwraps_grouped_conditions_under_compound():
    config = [
        {
            OrMarker.tag(): [
                {IntentDetectedMarker.tag(): ["1", "2"]},
                {IntentDetectedMarker.negated_tag(): ["3", "4", "5"]},
            ]
        },
    ]
    marker = Marker.from_config(config)
    assert isinstance(marker, OrMarker)
    assert len(marker.sub_markers) == 5
    assert all(
        isinstance(sub_marker, IntentDetectedMarker)
        for sub_marker in marker.sub_markers
    )
    assert set(sub_marker.text for sub_marker in marker.sub_markers) == {
        str(i + 1) for i in range(5)
    }


@pytest.mark.parametrize("marker_class", CONDITION_MARKERS)
def test_atomic_markers_negated_to_str(marker_class: Type[AtomicMarker]):
    marker = marker_class("intent1", negated=True)
    if marker.negated_tag() is not None:
        assert marker.negated_tag() in str(marker)


@pytest.mark.parametrize(
    "atomic_marker_type, negated", itertools.product(CONDITION_MARKERS, [False, True])
)
def test_atomic_marker_track(atomic_marker_type: Type[AtomicMarker], negated: bool):
    """Each marker applies an exact number of times (slots are immediately un-set)."""
    marker = atomic_marker_type(text="same-text", name="marker_name", negated=negated)
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


@pytest.mark.parametrize("atomic_marker_type", CONDITION_MARKERS)
def test_atomic_marker_evaluate_events(atomic_marker_type: Type[AtomicMarker]):
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
    marker = atomic_marker_type(text="same-text", name="marker_name")
    evaluation = marker.evaluate_events(events)
    assert len(evaluation) == 1
    assert "marker_name" in evaluation[0]
    if atomic_marker_type == IntentDetectedMarker:
        expected = [1, 3, 5]
    else:
        expected = [2, 4, 6]

    actual_preceeding_user_turns = [
        meta_data.preceding_user_turns for meta_data in evaluation[0]["marker_name"]
    ]
    assert actual_preceeding_user_turns == expected


@pytest.mark.parametrize("marker_class", OPERATOR_MARKERS)
def test_compound_markers_negated_to_str(marker_class: Type[CompoundMarker]):
    marker = marker_class([IntentDetectedMarker("bla")], negated=True)
    if marker.negated_tag() is not None:
        assert marker.negated_tag() in str(marker)


@pytest.mark.parametrize("negated", [True, False])
def test_compound_marker_or_track(negated: bool):
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
def test_compound_marker_and_track(negated: bool):
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


@pytest.mark.parametrize("negated", [True, False])
def test_compound_marker_seq_track(negated: bool):
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
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


@pytest.mark.parametrize("negated", [True, False])
def test_compound_marker_occur_track(negated: bool):
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (SlotSet("2", value="bla"), True),
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), True),
        (SlotSet("2", value=None), True),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), True),
    ]
    events, expected = zip(*events_expected)
    sub_markers = [IntentDetectedMarker("1"), SlotSetMarker("2")]
    marker = OccurrenceMarker(sub_markers, name="marker_name", negated=negated)
    for event in events:
        marker.track(event)
    expected = list(expected)
    if negated:
        expected = [not applies for applies in expected]
    assert marker.history == expected

    assert marker.relevant_events() == [expected.index(True)]


def test_compound_marker_occur_never_applied():
    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
    ]
    events, expected = zip(*events_expected)
    sub_markers = [IntentDetectedMarker("1"), SlotSetMarker("2")]
    marker = OccurrenceMarker(sub_markers, name="marker_name", negated=False)
    for event in events:
        marker.track(event)

    assert marker.relevant_events() == []


def test_compound_marker_nested_simple_track():
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
            OrMarker([IntentDetectedMarker("4"), IntentDetectedMarker("6"),]),
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
    possible_conditions: List[Type[AtomicMarker]],
    constant_negated: Optional[bool],
    possible_operators: List[Type[CompoundMarker]],
) -> Tuple[Marker, int]:
    """Generates an (max_branches)-ary tree with the specified depth."""
    if depth == 0:
        condition_class = possible_conditions[rng.choice(len(possible_conditions))]
        negated = bool(rng.choice(2)) if constant_negated is None else constant_negated
        condition_text = constant_condition_text or f"{rng.choice(1000)}"
        return condition_class(text=condition_text, negated=negated), 1
    else:
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
        operator_class = possible_operators[rng.choice(len(possible_operators))]
        negated = bool(rng.choice(2)) if constant_negated is None else constant_negated
        marker = operator_class(markers=sub_markers, negated=negated)
        marker_size += 1
        return marker, marker_size


@pytest.mark.parametrize(
    "depth, max_branches, seed", [(1, 3, 3456), (4, 3, 345), (4, 5, 2345)]
)
def test_compound_marker_nested_randomly_track(
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
    assert len([sub_marker for sub_marker in marker]) == expected_size
    for sub_marker in marker:
        assert len(sub_marker.history) == len(events)


@pytest.mark.parametrize(
    "depth, seed, matches",
    [
        (depth, seed, matches)
        for depth, seed in [(1, 3456), (4, 345)]
        for matches in [True, False]
    ],
)
def test_compound_marker_nested_randomly_ors_track(
    depth: int, seed: int, matches: bool
):
    rng = np.random.default_rng(seed=seed)
    constant_condition_text = "1" if matches else "2"
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
    if matches:
        assert all([any(sub_marker.history) for sub_marker in marker])
    else:
        assert all([not any(sub_marker.history) for sub_marker in marker])


def test_sessions_evaluated_separately():
    """Each marker applies an exact number of times (slots are immediately un-set)."""

    events = [
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
    assert evaluation[0]["my-marker"][0].idx == 3
    assert len(evaluation[1]["my-marker"]) == 1
    assert evaluation[1]["my-marker"][0].preceding_user_turns == 2
    assert evaluation[1]["my-marker"][0].idx == 7  # i.e. NOT the index in the dialogue


def test_atomic_markers_repr_not():
    marker = IntentDetectedMarker("intent1", negated=True)
    assert str(marker) == "(intent_not_detected: intent1)"


def test_all_operators_in_schema():
    operators_in_schema = rasa.shared.utils.schemas.markers.OPERATOR_SCHEMA["enum"]
    operators_in_schema = {tag.lower() for tag in operators_in_schema}

    actual_operators = set()
    for operator in OPERATOR_MARKERS:
        actual_operators.add(operator.tag())
        if operator.negated_tag():
            actual_operators.add(operator.negated_tag())

    assert actual_operators == operators_in_schema
