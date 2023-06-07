import textwrap

from rasa.core.policies.flow_policy import (
    FlowExecutor,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader
from rasa.shared.core.trackers import DialogueStateTracker


def test_select_next_action():
    flows = YAMLFlowsReader.read_from_string(
        textwrap.dedent(
            """
        flows:
          test_flow:
            description: Test flow
            steps:
              - id: "1"
                intent: transfer_money
                next: "2"
              - id: "2"
                action: utter_ask_name
        """
        )
    )
    tracker = DialogueStateTracker.from_dict(
        "test",
        [{"event": "user", "parse_data": {"intent": {"name": "transfer_money"}}}],
    )
    domain = Domain.empty()
    executor = FlowExecutor.from_tracker(tracker, flows)

    action, events, score = executor.advance_flows(tracker, domain)
    assert action == "flow_test_flow"
    assert events == []
    assert score == 1.0
