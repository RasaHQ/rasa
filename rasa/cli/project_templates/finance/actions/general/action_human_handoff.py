from typing import Any, Dict, List, Text

from openai import AsyncOpenAI
from rasa_sdk import Action, Tracker
from rasa_sdk.events import BotUttered
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionHumanHandoff(Action):
    def __init__(self) -> None:
        super().__init__()
        self.client = AsyncOpenAI()

    def name(self) -> Text:
        return "action_human_handoff"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        # Collect conversation
        convo = []
        for event in tracker.events:
            if event.get("event") == "user":
                user_text = "{}".format(event.get("text"))
                user_text2 = "user - " + user_text
                convo.append(user_text2)
            if event.get("event") == "bot":
                bot_text = "{}".format(event.get("text"))
                bot_text2 = "bot - " + bot_text
                # print(bot_text)
                convo.append(bot_text2)
        prompt = (
            f"The following is a conversation between a bot and a human user, "
            f"please summarise so that a human agent can easily understand "
            f"the important context. Conversation: {convo}"
        )
        response = await self.client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
        )
        summarised_conversation = response.choices[0].message.content or ""
        return [
            BotUttered(
                f"I will transfer the following summary of our conversation "
                f"to the Callback Manager:\n"
                f"{summarised_conversation}"
            )
        ]
