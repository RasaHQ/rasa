from rasa_core_sdk import Action
from rasa_core_sdk.events import SlotSet


class ActionSearchConcerts(Action):
    def name(self):
        return "action_search_concerts"

    def run(self, dispatcher, tracker, domain):
        concerts = [
            {"artist": "Foo Fighters", "reviews": 4.5},
            {"artist": "Katy Perry", "reviews": 5.0},
        ]
        description = ", ".join([c["artist"] for c in concerts])
        dispatcher.utter_message("{}".format(description))
        return [SlotSet("concerts", concerts)]


class ActionSearchVenues(Action):
    def name(self):
        return "action_search_venues"

    def run(self, dispatcher, tracker, domain):
        venues = [
            {"name": "Big Arena", "reviews": 4.5},
            {"name": "Rock Cellar", "reviews": 5.0},
        ]
        dispatcher.utter_message("here are some venues I found")
        description = ", ".join([c["name"] for c in venues])
        dispatcher.utter_message("{}".format(description))
        return [SlotSet("venues", venues)]


class ActionShowConcertReviews(Action):
    def name(self):
        return "action_show_concert_reviews"

    def run(self, dispatcher, tracker, domain):
        concerts = tracker.get_slot("concerts")
        dispatcher.utter_message("concerts from slots: {}".format(concerts))
        return []


class ActionShowVenueReviews(Action):
    def name(self):
        return "action_show_venue_reviews"

    def run(self, dispatcher, tracker, domain):
        venues = tracker.get_slot("venues")
        dispatcher.utter_message("venues from slots: {}".format(venues))
        return []
