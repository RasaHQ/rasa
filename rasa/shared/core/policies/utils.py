from typing import Set, Text

from rasa.dialogue_understanding.patterns.chitchat import FLOW_PATTERN_CHITCHAT
from rasa.graph_components.providers.forms_provider import Forms
from rasa.graph_components.providers.responses_provider import Responses
from rasa.shared.constants import (
    REQUIRED_SLOTS_KEY,
    UTTER_ASK_PREFIX,
    UTTER_FREE_CHITCHAT_RESPONSE,
)
from rasa.shared.core.constants import ACTION_TRIGGER_CHITCHAT
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.training_data.structures import StoryGraph


def collect_form_responses(forms: Forms) -> Set[Text]:
    """Collect responses that belong the requested slots in forms.

    Args:
        forms: the forms from the domain
    Returns:
        all utterances used in forms
    """
    form_responses = set()
    for _, form_info in forms.data.items():
        for required_slot in form_info.get(REQUIRED_SLOTS_KEY, []):
            form_responses.add(f"{UTTER_ASK_PREFIX}{required_slot}")
    return form_responses


def filter_responses_for_intentless_policy(
    responses: Responses, forms: Forms, flows: FlowsList
) -> Responses:
    """Filters out responses that are unwanted for the intentless policy.

    This includes utterances used in flows and forms.

    Args:
        responses: the responses from the domain
        forms: the forms from the domain
        flows: all flows
    Returns:
        The remaining, relevant responses for the intentless policy.
    """
    form_responses = collect_form_responses(forms)
    flow_responses = flows.utterances
    combined_responses = form_responses | flow_responses
    filtered_responses = {
        name: variants
        for name, variants in responses.data.items()
        if name not in combined_responses
    }

    pattern_chitchat = flows.flow_by_id(FLOW_PATTERN_CHITCHAT)

    # The following condition is highly unlikely, but mypy requires the case
    # of pattern_chitchat == None to be addressed
    if not pattern_chitchat:
        return Responses(data=filtered_responses)

    # if action_trigger_chitchat, filter out "utter_free_chitchat_response"
    has_action_trigger_chitchat = pattern_chitchat.has_action_step(
        ACTION_TRIGGER_CHITCHAT
    )
    if has_action_trigger_chitchat:
        filtered_responses.pop(UTTER_FREE_CHITCHAT_RESPONSE, None)

    return Responses(data=filtered_responses)


def contains_intentless_policy_responses(
    flows: FlowsList, domain: Domain, story_graph: StoryGraph
) -> bool:
    """Checks if IntentlessPolicy has applicable responses: either responses in the
    domain that are not part of any flow, or if there are e2e stories.
    """
    responses = filter_responses_for_intentless_policy(
        Responses(data=domain.responses), Forms(data=domain.forms), flows
    )

    has_applicable_responses = bool(
        responses and responses.data and len(responses.data) > 0
    )
    has_e2e_stories = bool(story_graph and story_graph.has_e2e_stories())

    return has_applicable_responses or has_e2e_stories
