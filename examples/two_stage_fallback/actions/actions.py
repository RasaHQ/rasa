# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import logging
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher
import csv

logger = logging.getLogger(__name__)


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "custom_fallback_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print(tracker.latest_message)
        dispatcher.utter_message(text="Please Contact Huawei Customer Care: URL")
        return [UserUtteranceReverted()]


class ActionAskAffirmation(Action):
    """Asks for an affirmation of the intent if NLU threshold is not met."""

    def name(self):
        return "action_default_ask_affirmation"

    def __init__(self):
        self.intent_mappings = {}
        # read the mapping from a csv and store it in a dictionary
        with open('intent_mapping.csv', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.intent_mappings[row[0]] = row[1]
        print(self.intent_mappings)

    def run(self, dispatcher, tracker, domain):
        # get the most likely intent
        intent1, intent2 = tracker.latest_message['intent_ranking'][:2]
        last_intent_name = tracker.latest_message['intent']['name']
        print(last_intent_name)
        # get the prompt for the intent
        intent1_prompt = self.intent_mappings[intent1['name']]
        intent2_prompt = self.intent_mappings[intent2['name']]
        logger.info(intent1_prompt, intent2_prompt)

        # Create the affirmation message and add two buttons to it.
        # Use '/<intent_name>' as payload to directly trigger '<intent_name>'
        # when the button is clicked.
        message = "I'm not sure if I understood you correctly. Do you mean..."
        buttons = [{'title': intent1_prompt,
                    'payload': '/{}'.format(intent1['name'])},
                   {'title': intent2_prompt,
                    'payload': '/{}'.format(intent2['name'])},
                   {'title': 'Something else!',
                    'payload': '/out_of_scope'}]
        dispatcher.utter_message(message, buttons=buttons)
        return []
