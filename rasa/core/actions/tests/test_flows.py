from rasa.core.actions.flows import FlowTriggerAction
from rasa.shared.core.events import SlotSet
from rasa.shared.core.flows.flow import START_STEP
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.constants import FLOW_STACK_SLOT
from rasa.shared.core.domain import Domain
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator


def test_name() -> None:
    action = FlowTriggerAction("flow_name")
    assert action.name() == "flow_name"
    assert action._flow_name == "name"


async def test_run_flow_action() -> None:
    action = FlowTriggerAction("flow_foobar")
    output_channel = OutputChannel()
    nlg = NaturalLanguageGenerator()
    tracker = DialogueStateTracker.from_dict("foobar", [])
    domain = Domain.empty()
    events = await action.run(output_channel, nlg, tracker, domain)
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == FLOW_STACK_SLOT
    assert events[0].value == [{"flow_id": "foobar", "step_id": START_STEP}]
