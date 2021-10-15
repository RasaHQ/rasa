from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet, ActionExecuted, UserUttered
from rasa.core.evaluation.markers_extract import Marker
from rasa.core.evaluation.markers_config import MarkerConfig


def test_marker_initializer():
    sample_yaml = """
    markers:
      - marker: no_restart
        operator: AND
        condition:
          - slot_set:
              - flight_class
          - slot_set:
              - travel_departure
              - travel_destination
    """
    config = MarkerConfig.from_yaml(sample_yaml)
    marker_dict = config.get("markers")[0]
    marker = Marker(
        name=marker_dict.get("marker"),
        operator=marker_dict.get("operator"),
        condition=marker_dict.get("condition"),
    )
    x = ["flight_class", "travel_departure", "travel_destination"]
    assert set(i for i in marker.slot_set) == set(i for i in x)


def test_markers_operator_or_slots():
    sample_yaml = """
        markers:
          - marker: marker_1
            operator: OR
            condition:
              - slot_set:
                  - flight_class
              - slot_not_set:
                  - travel_departure
                  - travel_destination
        """
    config = MarkerConfig.from_yaml(sample_yaml)
    marker_dict = config.get("markers")[0]
    marker = Marker(
        name=marker_dict.get("marker"),
        operator=marker_dict.get("operator"),
        condition=marker_dict.get("condition"),
    )

    domain = Domain.empty()
    tracker = DialogueStateTracker(sender_id="xyz", slots=None)
    tracker.update(SlotSet("travel_departure", value=None), domain)
    tracker.update(SlotSet("travel_departure", value="edinburgh"), domain)
    tracker.update(SlotSet("travel_departure", value="london"), domain)

    # check marker condition was satisfied
    assert marker.check_or(tracker.events)

    # check that it was only satisfied once at the end of the dialogue
    assert len(marker.timestamps) == 1
    assert marker.timestamps == [tracker.events[-1].timestamp]
    assert len(marker.preceding_user_turns) == 1
    assert marker.preceding_user_turns == [0]


def test_markers_operator_or_actions():
    sample_yaml = """
        markers:
          - marker: marker_1
            operator: OR
            condition:
              - action_executed:
                  - action_analyse_travelplan
              - action_not_executed:
                  - action_calculate_offsets
                  - action_disclaimer
        """
    config = MarkerConfig.from_yaml(sample_yaml)
    marker_dict = config.get("markers")[0]
    marker = Marker(
        name=marker_dict.get("marker"),
        operator=marker_dict.get("operator"),
        condition=marker_dict.get("condition"),
    )

    domain = Domain.empty()
    tracker = DialogueStateTracker(sender_id="xyz", slots=None)
    tracker.update(ActionExecuted("action_analyse_travelplan"), domain)
    tracker.update(ActionExecuted("action_analyse_travelplan"), domain)
    tracker.update(ActionExecuted("thank"), domain)

    # check marker condition was satisfied
    assert marker.check_or(tracker.events)

    # check that it eas satisfied at in the 1st and 2nd events,
    # then twice at the end for the two non executed actions
    timestamps = [e.timestamp for e in tracker.events]
    timestamps.append(tracker.events[-1].timestamp)
    assert len(marker.timestamps) == 4
    assert marker.timestamps == timestamps
    assert len(marker.preceding_user_turns) == 4
    assert marker.preceding_user_turns == [0, 0, 0, 0]


def test_markers_operator_or_intents():
    sample_yaml = """
        markers:
          - marker: marker_1
            operator: OR
            condition:
              - intent_detected:
                  - express_surprise
              - intent_not_detected:
                  - insult
                  - vulgar
        """
    config = MarkerConfig.from_yaml(sample_yaml)
    marker_dict = config.get("markers")[0]
    marker = Marker(
        name=marker_dict.get("marker"),
        operator=marker_dict.get("operator"),
        condition=marker_dict.get("condition"),
    )

    domain = Domain.empty()
    tracker = DialogueStateTracker(sender_id="xyz", slots=None)
    tracker.update(UserUttered(intent={"name": "insult"}), domain)
    tracker.update(UserUttered(intent={"name": "express_surprise"}), domain)
    tracker.update(UserUttered(intent={"name": "insult"}), domain)

    # check marker condition was satisfied
    assert marker.check_or(tracker.events)

    # check that it was satisfied at the second event
    assert len(marker.timestamps) == 2
    assert marker.timestamps == [
        tracker.events[1].timestamp,
        tracker.events[2].timestamp,
    ]
    assert len(marker.preceding_user_turns) == 2
    assert marker.preceding_user_turns == [2, 3]


def test_markers_operator_or_combo():
    sample_yaml = """
        markers:
          - marker: marker_1
            operator: OR
            condition:
              - slot_set:
                  - flight_class
              - action_executed:
                  - action_analyse_travelplan
              - intent_detected:
                  - express_surprise
              - slot_not_set:
                  - travel_departure
                  - travel_destination
              - action_not_executed:
                  - action_calculate_offsets
                  - action_disclaimer
              - intent_not_detected:
                  - insult
                  - vulgar
        """
    config = MarkerConfig.from_yaml(sample_yaml)
    marker_dict = config.get("markers")[0]
    marker = Marker(
        name=marker_dict.get("marker"),
        operator=marker_dict.get("operator"),
        condition=marker_dict.get("condition"),
    )

    domain = Domain.empty()
    tracker = DialogueStateTracker(sender_id="xyz", slots=None)
    tracker.update(SlotSet("travel_departure", value=None), domain)
    tracker.update(SlotSet("travel_departure", value="edinburgh"), domain)
    tracker.update(SlotSet("travel_departure", value="london"), domain)
    tracker.update(ActionExecuted("action_analyse_travelplan"), domain)  # here
    tracker.update(ActionExecuted("action_analyse_travelplan"), domain)  # here
    tracker.update(ActionExecuted("action_disclaimer"), domain)
    tracker.update(UserUttered(intent={"name": "insult"}), domain)
    tracker.update(UserUttered(intent={"name": "express_surprise"}), domain)  # here
    tracker.update(UserUttered(intent={"name": "insult"}), domain)

    # check marker condition was satisfied
    assert marker.check_or(tracker.events)

    # check that it was satisfied at the second event
    assert len(marker.timestamps) == 6
    assert marker.timestamps == [
        tracker.events[3].timestamp,
        tracker.events[4].timestamp,
        tracker.events[7].timestamp,
        tracker.events[-1].timestamp,
        tracker.events[-1].timestamp,
        tracker.events[-1].timestamp,
    ]
    assert len(marker.preceding_user_turns) == 6
    assert marker.preceding_user_turns == [0, 0, 2, 3, 3, 3]


def test_markers_operator_or_combo_none_found():
    sample_yaml = """
        markers:
          - marker: marker_1
            operator: OR
            condition:
              - slot_set:
                  - flight_class
              - action_executed:
                  - action_analyse_travelplan
              - intent_detected:
                  - express_surprise
        """
    config = MarkerConfig.from_yaml(sample_yaml)
    marker_dict = config.get("markers")[0]
    marker = Marker(
        name=marker_dict.get("marker"),
        operator=marker_dict.get("operator"),
        condition=marker_dict.get("condition"),
    )

    domain = Domain.empty()
    tracker = DialogueStateTracker(sender_id="xyz", slots=None)
    tracker.update(SlotSet("travel_departure", value=None), domain)
    tracker.update(ActionExecuted("action_disclaimer"), domain)
    tracker.update(UserUttered(intent={"name": "insult"}), domain)

    # check marker condition was satisfied
    assert not marker.check_or(tracker.events)

    # check that it was satisfied at the second event
    assert len(marker.timestamps) == 0
    assert marker.timestamps == []
    assert len(marker.preceding_user_turns) == 0
    assert marker.preceding_user_turns == []


def test_markers_operator_and():
    sample_yaml = """
        markers:
          - marker: marker_1
            operator: AND
            condition:
              - slot_set:
                  - flight_class
                  - travel_departure
              - action_executed:
                  - action_calculate_offsets
              - intent_not_detected:
                  - insult
                  - vulgar           
        """
    config = MarkerConfig.from_yaml(sample_yaml)
    marker_dict = config.get("markers")[0]
    marker = Marker(
        name=marker_dict.get("marker"),
        operator=marker_dict.get("operator"),
        condition=marker_dict.get("condition"),
    )

    domain = Domain.empty()
    tracker = DialogueStateTracker(sender_id="xyz", slots=None)
    tracker.update(SlotSet("flight_class", value="first"), domain)
    tracker.update(SlotSet("travel_departure", value="edinburgh"), domain)
    tracker.update(SlotSet("travel_destination", value="berlin"), domain)
    tracker.update(ActionExecuted("action_disclaimer"), domain)
    tracker.update(ActionExecuted("action_calculate_offsets"), domain)  # true
    tracker.update(SlotSet("flight_class", value="business"), domain)  # true
    tracker.update(SlotSet("travel_departure", value="berlin"), domain)  # true
    tracker.update(SlotSet("travel_destination", value="new york"), domain)  # true
    tracker.update(ActionExecuted("action_calculate_offsets"), domain)  # true

    assert marker.check_and(tracker.events)

    assert len(marker.timestamps) == 5
    assert len(marker.preceding_user_turns) == 5
