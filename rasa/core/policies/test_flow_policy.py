import textwrap

from rasa.core.policies.flow_policy import (
    FlowExecutor,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader
from rasa.shared.core.trackers import DialogueStateTracker


def _run_flow_until_listen(executor: FlowExecutor, tracker: DialogueStateTracker, domain: Domain):
    # Run the flow until we reach a listen action. collect and return all events and intermediate actions.
    events = []
    actions = []
    while True:
        action, new_events, _ = executor.advance_flows(tracker, domain)
        events.extend(new_events)
        actions.append(action)
        tracker.update_with_events(new_events, domain)
        if action:
            tracker.update(ActionExecuted(action), domain)
        if action == "action_listen":
            break
        if  action == None and len(new_events) == 0:
            # No action was executed and no events were generated. This means that the flow isn't 
            # doing anything anymore
            break
    return actions, events
    

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
        [
            {"event": "action", "name": "action_listen"},
            {"event": "user", "parse_data": {"intent": {"name": "transfer_money"}}}],
    )
    domain = Domain.empty()
    executor = FlowExecutor.from_tracker(tracker, flows)

    actions, events = _run_flow_until_listen(executor, tracker, domain)

    assert actions == ["flow_test_flow", None]
    assert events == []

