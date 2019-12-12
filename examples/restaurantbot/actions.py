from rasa_sdk import Action
from rasa_sdk.events import SlotSet


class RestaurantAPI:
    def search(self, info):
        return "papi's pizza place"


class ActionSearchRestaurants(Action):
    def name(self):
        return "action_search_restaurants"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="looking for restaurants")
        restaurant_api = RestaurantAPI()
        restaurants = restaurant_api.search(tracker.get_slot("cuisine"))
        return [SlotSet("matches", restaurants)]


class ActionSuggest(Action):
    def name(self):
        return "action_suggest"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="here's what I found:")
        dispatcher.utter_message(text=tracker.get_slot("matches"))
        dispatcher.utter_message(
            text="is it ok for you? hint: I'm not going to find anything else :)"
        )
        return []
