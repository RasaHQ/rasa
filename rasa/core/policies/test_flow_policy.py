import textwrap
from typing import List, Optional, Text, Tuple

from rasa.core.policies.flow_policy import (
    FlowExecutor,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, Event
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader
from rasa.shared.core.trackers import DialogueStateTracker


def _run_flow_until_listen(
    executor: FlowExecutor, tracker: DialogueStateTracker, domain: Domain
) -> Tuple[List[Optional[Text]], List[Event]]:
    # Run the flow until we reach a listen action.
    # Collect and return all events and intermediate actions.
    events = []
    actions = []
    while True:
        action_prediction = executor.advance_flows(tracker)
        if not action_prediction:
            break

        events.extend(action_prediction.events or [])
        actions.append(action_prediction.action_name)
        tracker.update_with_events(action_prediction.events or [], domain)
        if action_prediction.action_name:
            tracker.update(ActionExecuted(action_prediction.action_name), domain)
        if action_prediction.action_name == "action_listen":
            break
        if action_prediction.action_name is None and not action_prediction.events:
            # No action was executed and no events were generated. This means that
            # the flow isn't doing anything anymore
            break
    return actions, events


def test_select_next_action() -> None:
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
            {"event": "user", "parse_data": {"intent": {"name": "transfer_money"}}},
        ],
    )
    domain = Domain.empty()
    executor = FlowExecutor.from_tracker(tracker, flows, domain)

    actions, events = _run_flow_until_listen(executor, tracker, domain)

    assert actions == ["flow_test_flow", None]
    assert events == []
