from rasa.core.evaluation.marker_tracker_loader import MarkerTrackerLoader
from rasa.core.tracker_store import TrackerStore
import pytest


# Assuming a TrackerStore with 5 trackers in it


def test_load_sample(tracker_store: TrackerStore):
    loader = MarkerTrackerLoader(tracker_store, "sample", 3)
    result = loader.load()

    assert len(result) == 3

    for item in result:
        assert tracker_store.exists(item.sender_id)


def test_load_first_n(tracker_store: TrackerStore):
    loader = MarkerTrackerLoader(tracker_store, "first_n", 3)
    result = loader.load()

    assert len(result) == 3

    for item in result:
        assert tracker_store.exists(item.sender_id)


def test_load_all(tracker_store: TrackerStore):
    loader = MarkerTrackerLoader(tracker_store, "all")
    result = loader.load()

    assert len(result) == len(tracker_store.keys())

    for item in result:
        assert tracker_store.exists(item.sender_id)


def test_exception_invalid_strategy(tracker_store: TrackerStore):
    with pytest.raises(Exception):  # Make this more specific
        loader = MarkerTrackerLoader(tracker_store, "summon")


def test_exception_no_count(tracker_store: TrackerStore):
    with pytest.raises(Exception):  # Make this more specific
        loader = MarkerTrackerLoader(tracker_store, "sample")


# TBD
def test_warn_seed_unnecessary(tracker_store: TrackerStore):
    loader = MarkerTrackerLoader(tracker_store, "first_n", 3, seed=5)


def test_warn_count_all_unnecessary(tracker_store: TrackerStore):
    loader = MarkerTrackerLoader(tracker_store, "all", 3)


def test_warn_count_exceeds_store(tracker_store: TrackerStore):
    loader = MarkerTrackerLoader(tracker_store, "sample", 6)
