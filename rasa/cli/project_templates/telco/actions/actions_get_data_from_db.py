import csv
from typing import Dict, List

from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.interfaces import Tracker
from rasa_sdk.types import DomainDict


class ActionGetCustomerInfo(Action):
    def name(self) -> str:
        return "action_get_customer_info"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict]:
        # Load CSV file
        file_path = "csvs/customers.csv"  # get information from your DBs
        customer_id = tracker.get_slot("customer_id")

        try:
            with open(file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                # Filter data for the given customer ID
                customer_info = None
                for row in reader:
                    if row["customer_id"] == str(customer_id):
                        customer_info = row
                        break

                if customer_info is None:
                    dispatcher.utter_message("No customer found with this ID.")
                    return []

                # Extract customer details
                first_name = customer_info["first_name"]

                # Set the retrieved name in a slot
                return [SlotSet("customer_first_name", first_name)]

        except FileNotFoundError:
            dispatcher.utter_message("Customer database file not found.")
            return []
        except Exception:
            dispatcher.utter_message("Error retrieving customer information.")
            return []
