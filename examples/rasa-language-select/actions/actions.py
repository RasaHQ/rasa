# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import json
import random
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import fasttext


class LanguageIdentification:
    def __init__(self):
        pretrained_lang_model = "lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text)  # returns top 2 matching languages
        return predictions[0][0].split('__')[-1]


class MultilingualResponse:
    def __init__(self):
        with open("multilingual_response.json") as fp:
            self.multilingual_response = json.load(fp)

    @staticmethod
    def random_response(responses):
        num_response = len(responses) - 1
        print(num_response)
        return responses[random.randint(0, num_response)]

    def predict_response(self, intent, lang):
        responses = self.multilingual_response[intent][lang]  # returns top 2 matching languages
        return self.random_response(responses)


multilingual_response = MultilingualResponse()
language_detection = LanguageIdentification()


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_utter_language_select"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print(tracker.latest_message)
        intent = tracker.latest_message['intent'].get('name')
        input_text = tracker.latest_message['text']
        lang = language_detection.predict_lang(input_text)
        response = multilingual_response.predict_response(intent=intent, lang=lang)
        print(lang, intent, response)
        dispatcher.utter_message(text=response)
        return []
