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

    assert marker.check_or(tracker.events)


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
    tracker.update(ActionExecuted("action_disclaimer"), domain)

    assert marker.check_or(tracker.events)


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

    assert marker.check_or(tracker.events)


def test_markers_operator_or_combo():
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
              - action_executed:
                  - action_analyse_travelplan
              - action_not_executed:
                  - action_calculate_offsets
                  - action_disclaimer
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
    tracker.update(SlotSet("travel_departure", value=None), domain)
    tracker.update(SlotSet("travel_departure", value="edinburgh"), domain)
    tracker.update(SlotSet("travel_departure", value="london"), domain)
    tracker.update(ActionExecuted("action_analyse_travelplan"), domain)
    tracker.update(ActionExecuted("action_analyse_travelplan"), domain)
    tracker.update(ActionExecuted("action_disclaimer"), domain)
    tracker.update(UserUttered(intent={"name": "insult"}), domain)
    tracker.update(UserUttered(intent={"name": "express_surprise"}), domain)
    tracker.update(UserUttered(intent={"name": "insult"}), domain)

    assert marker.check_or(tracker.events)
