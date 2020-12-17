import copy
import time
from typing import List, Text, Optional

from rasa.core.actions import action
from rasa.core.actions.loops import LoopAction
from rasa.core.channels import OutputChannel
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    Event,
    UserUtteranceReverted,
    ActionExecuted,
    UserUttered,
    ActiveLoop,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.shared.core.constants import (
    USER_INTENT_OUT_OF_SCOPE,
    ACTION_LISTEN_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
)
from rasa.utils.endpoints import EndpointConfig


class TwoStageFallbackAction(LoopAction):
    def __init__(self, action_endpoint: Optional[EndpointConfig] = None) -> None:
        self._action_endpoint = action_endpoint

    def name(self) -> Text:
        return ACTION_TWO_STAGE_FALLBACK_NAME

    async def do(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        if _user_should_affirm(tracker, events_so_far):
            return await self._ask_affirm(output_channel, nlg, tracker, domain)

        return await self._ask_rephrase(output_channel, nlg, tracker, domain)

    async def _ask_affirm(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> List[Event]:
        affirm_action = action.action_for_name_or_text(
            ACTION_DEFAULT_ASK_AFFIRMATION_NAME, domain, self._action_endpoint
        )

        return await affirm_action.run(output_channel, nlg, tracker, domain)

    async def _ask_rephrase(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> List[Event]:
        rephrase = action.action_for_name_or_text(
            ACTION_DEFAULT_ASK_REPHRASE_NAME, domain, self._action_endpoint
        )

        return await rephrase.run(output_channel, nlg, tracker, domain)

    async def is_done(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> bool:
        _user_clarified = _last_intent_name(tracker) not in [
            DEFAULT_NLU_FALLBACK_INTENT_NAME,
            USER_INTENT_OUT_OF_SCOPE,
        ]
        return (
            _user_clarified
            or _two_fallbacks_in_a_row(tracker)
            or _second_affirmation_failed(tracker)
        )

    async def deactivate(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        if _two_fallbacks_in_a_row(tracker) or _second_affirmation_failed(tracker):
            return await self._give_up(output_channel, nlg, tracker, domain)

        # revert fallback events
        return [UserUtteranceReverted()] + _message_clarification(tracker)

    async def _give_up(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> List[Event]:
        fallback = action.action_for_name_or_text(
            ACTION_DEFAULT_FALLBACK_NAME, domain, self._action_endpoint
        )

        return await fallback.run(output_channel, nlg, tracker, domain)


def _last_intent_name(tracker: DialogueStateTracker) -> Optional[Text]:
    last_message = tracker.latest_message
    if not last_message:
        return None

    return last_message.intent.get("name")


def _two_fallbacks_in_a_row(tracker: DialogueStateTracker) -> bool:
    return _last_n_intent_names(tracker, 2) == [
        DEFAULT_NLU_FALLBACK_INTENT_NAME,
        DEFAULT_NLU_FALLBACK_INTENT_NAME,
    ]


def _last_n_intent_names(
    tracker: DialogueStateTracker, number_of_last_intent_names: int
) -> List[Text]:
    intent_names = []
    for i in range(number_of_last_intent_names):
        message = tracker.get_last_event_for(
            (UserUttered, UserUtteranceReverted),
            skip=i,
            event_verbosity=EventVerbosity.AFTER_RESTART,
        )
        if isinstance(message, UserUttered):
            intent_names.append(message.intent.get("name"))

    return intent_names


def _user_should_affirm(
    tracker: DialogueStateTracker, events_so_far: List[Event]
) -> bool:
    fallback_was_just_activated = any(
        isinstance(event, ActiveLoop) for event in events_so_far
    )
    if fallback_was_just_activated:
        return True

    return _last_intent_name(tracker) == DEFAULT_NLU_FALLBACK_INTENT_NAME


def _second_affirmation_failed(tracker: DialogueStateTracker) -> bool:
    return _last_n_intent_names(tracker, 3) == [
        USER_INTENT_OUT_OF_SCOPE,
        DEFAULT_NLU_FALLBACK_INTENT_NAME,
        USER_INTENT_OUT_OF_SCOPE,
    ]


def _message_clarification(tracker: DialogueStateTracker) -> List[Event]:
    clarification = copy.deepcopy(tracker.latest_message)
    clarification.parse_data["intent"]["confidence"] = 1.0
    clarification.timestamp = time.time()
    return [ActionExecuted(ACTION_LISTEN_NAME), clarification]
