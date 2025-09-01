from typing import Any, Dict, List, Text

import openai
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionHumanHandoff(Action):
    def name(self) -> Text:
        return "action_human_handoff"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        convo: List[str] = []
        for event in tracker.events:
            if event.get("event") == "user":
                user_text = str(event.get("text") or "")
                convo.append(f"user - {user_text}")
            elif event.get("event") == "bot":
                bot_text = str(event.get("text") or "")
                convo.append(f"bot - {bot_text}")
        prompt = (
            f"The following is a conversation between a bot and a human user. "
            f"Please summarise so that a human agent can easily understand the "
            f"important context. Conversation: "
            f"{convo}"
        )
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        summarised_conversation = (
            response.choices[0].message.content or "No summary available"
        )
        dispatcher.utter_message(
            response="utter_transfer_to_manager", summary=summarised_conversation
        )
        return []
