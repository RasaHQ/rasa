from typing import Any, Dict, List, Text

import openai
from rasa_sdk import Action, Tracker
from rasa_sdk.events import BotUttered
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionHumanHandoff(Action):
    def name(self) -> Text:
        return "action_human_handoff"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        # Collect conversation
        convo = []
        for event in tracker.events:
            if event.get("event") == "user":
                user_text = f"user - {event.get('text')}"
                convo.append(user_text)
            if event.get("event") == "bot":
                bot_text = f"bot - {event.get('text')}"
                convo.append(bot_text)
        prompt = (
            f"The following is a conversation between a bot and a human user, "
            f"please summarise so that a human agent can easily understand "
            f"the important context. Conversation: {convo}"
        )
        response = openai.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
        )
        summarised_conversation = (
            response.choices[0].message.content or "No summary available"
        )
        return [
            BotUttered(
                f"I will transfer the following summary of our conversation "
                f"to the Callback Manager:\n"
                f"{summarised_conversation}"
            )
        ]
