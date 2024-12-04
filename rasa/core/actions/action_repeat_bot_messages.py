from typing import Optional, Dict, Any, List

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
from rasa.shared.core.events import Event, BotUttered, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


class ActionRepeatBotMessages(Action):
    """Action to repeat bot messages"""

    def name(self) -> str:
        """Return the name of the action."""
        return ACTION_REPEAT_BOT_MESSAGES

    def _get_last_bot_events(self, tracker: DialogueStateTracker) -> List[Event]:
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
        # Skip action if we are in a collect information step whose
        # default behavior is to repeat anyways
        top_frame = tracker.stack.top(
            lambda frame: isinstance(frame, RepeatBotMessagesPatternFlowStackFrame)
            or isinstance(frame, UserSilencePatternFlowStackFrame)
        )
        if isinstance(top_frame, CollectInformationPatternFlowStackFrame):
            return []
        # filter user and bot events
        filtered = [
            e for e in tracker.events if isinstance(e, (BotUttered, UserUttered))
        ]
        bot_events: List[Event] = []

        # find the last BotUttered events
        for e in reversed(filtered):
            if isinstance(e, BotUttered):
                # insert instead of append because the list is reversed
                bot_events.insert(0, e)

            # stop if a UserUttered event is found
            # only if we have collected some bot events already
            # this condition skips the first N UserUttered events
            elif bot_events:
                break

        return bot_events

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Send the last bot messages to the channel again"""
        bot_events = self._get_last_bot_events(tracker)
        return bot_events
