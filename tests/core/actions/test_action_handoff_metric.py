from rasa.core.actions.action_handoff_metric import ActionHandoffMetric
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.constants import ACTION_HANDOFF_METRIC
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


def test_ACTION_HANDOFF_METRIC():
    """ActionHandoffMetric returns the correct system action name."""
    action = ActionHandoffMetric()
    assert action.name() == ACTION_HANDOFF_METRIC


async def test_action_handoff_metric_returns_no_events():
    """Test that action is a no-op and returns no events."""
    tracker = DialogueStateTracker.from_events(
        "test_sender",
        [UserUttered("I need to speak to a human")],
    )
    domain = Domain.empty()
    action = ActionHandoffMetric()
    channel = CollectingOutputChannel()
    nlg = NaturalLanguageGenerator()

    events = await action.run(channel, nlg, tracker, domain)

    assert events == []
    assert len(channel.messages) == 0
