import asyncio

from rasa_sdk import Action
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.interfaces import Tracker
from rasa_sdk.types import DomainDict


class ActionSleepAndRespond(Action):
    def name(self) -> str:
        return "action_sleep_few_sec"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> list[EventType]:
        await asyncio.sleep(3)
        return []
