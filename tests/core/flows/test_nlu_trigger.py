import pytest

from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    PREDICTED_CONFIDENCE_KEY,
    INTENT_NAME_KEY,
)
from rasa.shared.nlu.training_data.message import Message


def test_valid_flow_with_nlu_triggers():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            nlu_trigger:
              - intent: foo
              - intent:
                  name: bar
                  confidence_threshold: 0.5
              - intent:
                  name: foobar
            steps:
              - action: utter_welcome
        """
    )
    flow = all_flows.flow_by_id("foo")
    trigger_conditions = flow.nlu_triggers.trigger_conditions

    assert len(trigger_conditions) == 3

    assert trigger_conditions[0].intent == "foo"
    assert trigger_conditions[0].confidence_threshold == 0.0
    assert trigger_conditions[1].intent == "bar"
    assert trigger_conditions[1].confidence_threshold == 0.5
    assert trigger_conditions[2].intent == "foobar"
    assert trigger_conditions[2].confidence_threshold == 0.0


@pytest.mark.parametrize(
    "intent, is_triggered",
    [
        (
            {
                INTENT_NAME_KEY: "bar",
                PREDICTED_CONFIDENCE_KEY: 1.0,
            },
            True,
        ),
        (
            {
                INTENT_NAME_KEY: "bar",
                PREDICTED_CONFIDENCE_KEY: 0.2,
            },
            False,
        ),
        (
            {
                INTENT_NAME_KEY: "not_a_trigger",
                PREDICTED_CONFIDENCE_KEY: 1.0,
            },
            False,
        ),
        (
            {
                INTENT_NAME_KEY: "bar",
                PREDICTED_CONFIDENCE_KEY: 0.5,
            },
            True,
        ),
    ],
)
def test_nlu_trigger_is_triggered(intent, is_triggered):
    all_flows = flows_from_str(
        """
        flows:
          foo:
            nlu_trigger:
              - intent:
                  name: bar
                  confidence_threshold: 0.5
            steps:
              - action: utter_welcome
        """
    )

    flow_with_nlu_trigger = all_flows.underlying_flows[0]

    message = Message(
        data={
            TEXT: "This is a test sentence.",
            INTENT: intent,
        }
    )

    assert is_triggered == flow_with_nlu_trigger.nlu_triggers.is_triggered(message)
