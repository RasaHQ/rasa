from rasa_core_sdk.forms import FormAction


class RestaurantForm(FormAction):

    def name(self):
        return "restaurant_form"

    @staticmethod
    def required_slots():
        return ["cuisine", "num_people"]

    def submit(self, dispatcher, tracker, domain):
        dispatcher.utter_message("done!")
        return []
