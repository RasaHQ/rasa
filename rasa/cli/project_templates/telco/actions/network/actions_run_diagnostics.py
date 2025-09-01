import asyncio
import random

from rasa_sdk import Action
from rasa_sdk.events import EventType, SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.interfaces import Tracker
from rasa_sdk.types import DomainDict


class ActionSleepAndRespond(Action):
    def name(self) -> str:
        return "actions_run_speed_test"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> list[EventType]:
        await asyncio.sleep(10)
        dispatcher.utter_message(
            text="Thank you for waiting... ✅ "
        )  # Send to the user

        random_number = random.randint(
            50, 150
        )  # we will pick randomly the internet speed here
        # for local testing purposes you can define the number
        # for Production testing connect to the API to get this data
        return [SlotSet("network_speed", random_number)]
