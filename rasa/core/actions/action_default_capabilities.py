"""Built-in action that dynamically generates a capabilities' response.

Answers "what can you do?" by enumerating the flows that are startable for the
current conversation (guard conditions evaluated against current tracker state).
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text

from rasa.core.actions.action import Action, create_bot_utterance
from rasa.shared.core.constants import ACTION_DEFAULT_CAPABILITIES_NAME
from rasa.shared.core.events import Event

if TYPE_CHECKING:
    from rasa.core.channels.channel import OutputChannel
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.flows.flows_list import FlowsList
    from rasa.shared.core.trackers import DialogueStateTracker

_NO_CAPABILITIES_TEXT = "I don't have any specific capabilities available right now."


async def _maybe_rephrase(
    response_dict: Dict[str, Any],
    nlg: "NaturalLanguageGenerator",
    tracker: "DialogueStateTracker",
    output_channel: "OutputChannel",
) -> Dict[str, Any]:
    """Pass the response through the contextual rephraser if one is configured.

    The rephraser is only invoked when the NLG endpoint is configured with
    ``type: rephrase`` in endpoints.yml -- the same condition that activates
    rephrasing for regular ``utter_`` responses with ``metadata: {rephrase: true}``.
    """
    from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser

    if isinstance(nlg, ContextualResponseRephraser):
        return await nlg.rephrase(response_dict, tracker, output_channel)
    return response_dict


def _render_capabilities(flow_entries: List[Dict[str, Any]]) -> Text:
    """Render a bullet-list of capabilities from serialized flow entries."""
    lines = ["Here's what I can help you with:"]
    for entry in flow_entries:
        name = entry["name"]
        description = entry.get("description") or ""
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


class ActionDefaultCapabilities(Action):
    """Dynamically generates a capabilities response from the loaded FlowsList.

    Only flows whose guard conditions are satisfied for the current conversation
    are included, so the response stays accurate as context changes.

    Customers who need custom formatting or richer output (e.g. building a UI
    payload) should override this action by defining a custom action with the
    same name ``action_default_capabilities`` in their action server.
    """

    def name(self) -> Text:
        return ACTION_DEFAULT_CAPABILITIES_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
        flows: Optional["FlowsList"] = None,
    ) -> List[Event]:
        """Generate and utter a capabilities' response.

        Args:
            output_channel: The output channel to send messages to.
            nlg: The natural language generator.
            tracker: Current conversation state used to evaluate guard conditions.
            domain: The domain.
            metadata: Standard action metadata (unused by this action).
            flows: Live ``FlowsList`` injected by the processor via the
                ``"flows" in run_args`` inspection pattern. When ``None``
                (e.g. non-CALM assistants), a fallback message is returned.

        Returns:
            A list containing a single ``BotUttered`` event.
        """
        if not flows:
            return [create_bot_utterance({"text": _NO_CAPABILITIES_TEXT})]

        startable = tracker.get_startable_flows(flows.user_flows)

        if not startable.underlying_flows:
            return [create_bot_utterance({"text": _NO_CAPABILITIES_TEXT})]

        entries = [
            {
                "name": f.custom_name or f.id,
                "description": f.description,
            }
            for f in startable.underlying_flows
            # exclude user flow triggering this default action
            # from composing assistant dynamic capabilities
            if f.id != tracker.active_flow
        ]

        if not entries:
            return [create_bot_utterance({"text": _NO_CAPABILITIES_TEXT})]

        response_dict: Dict[str, Any] = {"text": _render_capabilities(entries)}
        response_dict = await _maybe_rephrase(
            response_dict, nlg, tracker, output_channel
        )
        return [create_bot_utterance(response_dict)]
