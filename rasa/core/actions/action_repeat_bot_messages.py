from typing import Any, Dict, List, Optional

from rasa.core.actions.action import Action
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.repeat import (
    RepeatBotMessagesPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.user_silence import (
    UserSilencePatternFlowStackFrame,
)
from rasa.shared.core.constants import ACTION_REPEAT_BOT_MESSAGES
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, Event, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


class ActionRepeatBotMessages(Action):
    """Action to repeat bot messages"""

    def name(self) -> str:
        """Return the name of the action."""
        return ACTION_REPEAT_BOT_MESSAGES

    def _get_last_bot_events(self, tracker: DialogueStateTracker) -> List[BotUttered]:
        """Get the last consecutive bot events before the most recent user message.

        This function scans the dialogue history in reverse to find the last sequence of
        bot responses that occurred without any user interruption. It filters out all
        non-utterance events and stops when it encounters a user message after finding
        bot messages.

        Args:
            tracker: DialogueStateTracker containing the conversation events.

        Returns:
            List[Event]: A list of consecutive BotUttered events that occurred
                most recently, in chronological order. Returns an empty list
                if no bot messages are found or if the last message was from
                the user.

        Example:
            For events: [User1, Bot1, Bot2, User2, Bot4, Bot5, User3]
            Returns: [Bot4, Bot5] (the last two bot events)
            The elif condition doesn't break when it sees User3 event.
            But it does at User2 event.
        """
        # filter user and bot events
        user_and_bot_events = [
            e for e in tracker.events if isinstance(e, (BotUttered, UserUttered))
        ]
        last_bot_events: List[BotUttered] = []

        # find the last BotUttered events
        for e in reversed(user_and_bot_events):
            # stop when seeing a user event after having seen bot events already
            if isinstance(e, UserUttered) and len(last_bot_events) > 0:
                break
            elif isinstance(e, BotUttered):
                last_bot_events.append(e)

        return list(reversed(last_bot_events))

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Send the last bot messages to the channel again"""
        top_frame = tracker.stack.top(
            lambda frame: isinstance(frame, RepeatBotMessagesPatternFlowStackFrame)
            or isinstance(frame, UserSilencePatternFlowStackFrame)
        )

        bot_events: List[Event] = list(self._get_last_bot_events(tracker))
        # drop the last bot event in a collect step as that part will be repeated anyway
        if isinstance(top_frame, CollectInformationPatternFlowStackFrame):
            bot_events = bot_events[:-1]
        return bot_events
