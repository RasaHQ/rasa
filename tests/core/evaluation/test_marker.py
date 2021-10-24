from typing import Type

import pytest

from rasa.core.evaluation.marker import (
    ActionExecutedMarker,
    AndMarker,
    IntentDetectedMarker,
    OrMarker,
    SlotSetMarker,
    SequenceMarker,
)
from rasa.core.evaluation.marker_base import Marker, AtomicMarker

from rasa.shared.core.events import SlotSet, ActionExecuted, UserUttered

from rasa.shared.nlu.constants import INTENT_NAME_KEY


EVENT_MARKERS = [ActionExecutedMarker, SlotSetMarker, IntentDetectedMarker]
COMPOUND_MARKERS = [AndMarker, OrMarker, SequenceMarker]


def test_marker_from_config_dict_single_and():

    config = {
        "marker_1": {
            "and": [
                {"slot_set": ["s1"]},
                {"or": [{"intent_detected": ["4"]}, {"intent_detected": ["6"]},]},
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
        {"slot_set": ["s1"]},
        {"or": [{"intent_detected": ["4"]}, {"intent_detected": ["6"]},]},
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
            "or": [
                {"intent_detected": ["1", "2"]},
                {"intent_detected": ["3", "4", "5"]},
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


@pytest.mark.parametrize("atomic_marker_type", EVENT_MARKERS)
def test_atomic_marker_track(atomic_marker_type: Type[AtomicMarker]):
    """Each marker applies an exact number of times (slots are immediately un-set)."""

    marker = atomic_marker_type(text="same-text", name="marker_name")

    events = [
        UserUttered(intent={"name": "1"}),
        UserUttered(intent={"name": "same-text"}),
        SlotSet("same-text", value="any"),
        SlotSet("same-text", value=None),
        ActionExecuted(action_name="same-text"),
    ]

    num_applies = 3
    events = events * num_applies

    for event in events:
        marker.track(event)

    assert len(marker.history) == len(events)
    assert sum(marker.history) == num_applies


@pytest.mark.parametrize("atomic_marker_type", EVENT_MARKERS)
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

    assert "marker_name" in evaluation
    if atomic_marker_type == IntentDetectedMarker:
        expected = [1, 3, 5]
    else:
        expected = [2, 4, 6]

    assert evaluation["marker_name"]["preceeding_user_turns"] == expected


def test_compound_marker_or_track():

    events = [
        UserUttered(intent={INTENT_NAME_KEY: "1"}),
        UserUttered(intent={INTENT_NAME_KEY: "unknown"}),
        UserUttered(intent={INTENT_NAME_KEY: "2"}),
        UserUttered(intent={INTENT_NAME_KEY: "unknown"}),
    ]

    sub_markers = [IntentDetectedMarker("1"), IntentDetectedMarker("2")]
    marker = OrMarker(sub_markers, name="marker_name")
    for event in events:
        marker.track(event)

    assert marker.history == [True, False, True, False]


def test_compound_marker_and_track():

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
    marker = AndMarker(sub_markers, name="marker_name")
    for event in events:
        marker.track(event)

    assert marker.history == list(expected)


def test_compound_marker_seq_track():

    events_expected = [
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), True),
        (UserUttered(intent={INTENT_NAME_KEY: "3"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "1"}), False),
        (UserUttered(intent={INTENT_NAME_KEY: "2"}), True),
    ]
    events, expected = zip(*events_expected)

    sub_markers = [IntentDetectedMarker("1"), IntentDetectedMarker("2")]
    marker = SequenceMarker(sub_markers, name="marker_name")
    for event in events:
        marker.track(event)

    assert marker.history == list(expected)


def test_compound_marker_nested_track():

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

    assert evaluation["marker_name"]["preceeding_user_turns"] == [3, 5]


def test_atomic_markers_repr_not():
    marker = IntentDetectedMarker("intent1", negated=True)
    assert str(marker) == "(intent_not_detected: intent1)"
