import csv
import logging
from datetime import datetime
from typing import Any, Dict, List

from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.interfaces import Tracker
from rasa_sdk.types import DomainDict


class ActionVerifyBillByDate(Action):
    def name(self) -> str:
        return "action_verify_bill_by_date"

    @staticmethod
    def text_to_date(month_text: str) -> str:
        try:
            # Get the current year
            current_year = datetime.now().year

            # Combine user input with the current year
            full_text = f"{month_text} {current_year}"

            # Parse the text format (e.g., "March 2025")
            date_obj = datetime.strptime(full_text, "%B %Y")

            # Format as DD/MM/YYYY (defaults to the first day of the month)
            formatted_date = date_obj.strftime("01/%m/%Y")
            logging.info(f"This is an info message: formatted_date: {formatted_date}")
            return formatted_date
        except ValueError:
            return "Invalid format. Please use a full month name (e.g., 'March')."

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[str, Any]]:
        # Get customer ID and date from slots
        customer_id = tracker.get_slot("customer_id")
        bill_month = tracker.get_slot("bill_month")

        bill_date = ActionVerifyBillByDate.text_to_date(bill_month)

        if not customer_id:
            dispatcher.utter_message(
                "I couldn't find your customer ID. Please provide it."
            )
            return []

        if not bill_date:
            dispatcher.utter_message(
                "Please specify the date for the bill you want to check."
            )
            return []

        try:
            # Load CSV file with billing data
            with open("csvs/billing.csv", "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                # Convert bill_date to datetime for comparison
                bill_date_obj = datetime.strptime(bill_date, "%d/%m/%Y")

                # Filter data for the given customer and date
                customer_bills = []
                specific_bill = None

                for row in reader:
                    if row["customer_id"] == str(customer_id):
                        # Parse the date from CSV
                        row_date = datetime.strptime(row["date"], "%d/%m/%Y")
                        row["amount"] = float(row["amount"])
                        customer_bills.append(row)

                        # Check if this is the specific bill we're looking for
                        if row_date.date() == bill_date_obj.date():
                            specific_bill = row

                if specific_bill is None:
                    dispatcher.utter_message(
                        f"No bill found for {bill_date_obj.date()}."
                    )
                    return []

                bill_amount = float(specific_bill["amount"])

                # Calculate average
                if customer_bills:
                    average_bill = sum(
                        float(bill["amount"]) for bill in customer_bills
                    ) / len(customer_bills)
                else:
                    average_bill = 0.0

                difference = bill_amount - average_bill

                # Generate response
                response = (
                    f"Your bill for {bill_month} {bill_date_obj.date().year} is "
                    f"${bill_amount:.2f}. \n"
                    f"The average of your past bills is ${average_bill:.2f}. \n"
                    f"This bill is {'higher' if difference > 0 else 'lower'} than "
                    f"your average by ${abs(difference):.2f}."
                )

                dispatcher.utter_message(response)
                return [
                    SlotSet("bill_amount", float(bill_amount)),
                    SlotSet("average_bill", float(average_bill)),
                    SlotSet("difference", float(difference)),
                ]

        except FileNotFoundError:
            dispatcher.utter_message("Billing data file not found.")
            return []
        except Exception:
            dispatcher.utter_message("Error retrieving billing information.")
            return []


class ActionRecapBill(Action):
    def name(self) -> str:
        return "action_recap_bill"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[str, Any]]:
        # Get customer_id and bill_date from slots
        customer_id = tracker.get_slot("customer_id")
        bill_month = tracker.get_slot("bill_month")

        bill_date = ActionVerifyBillByDate.text_to_date(bill_month)

        if not customer_id:
            dispatcher.utter_message(
                "I need your customer ID to fetch your bill recap."
            )
            return []

        if not bill_date:
            dispatcher.utter_message(
                "I need a date to fetch your bill recap. Can you provide one?"
            )
            return []

        # Convert customer_id to int if needed
        try:
            customer_id = int(customer_id)
        except ValueError:
            dispatcher.utter_message("Invalid customer ID format.")
            return []

        try:
            # Load CSV file
            with open("csvs/billing.csv", "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                bill_date_obj = datetime.strptime(bill_date, "%d/%m/%Y")

                # Filter records for the given customer_id
                filtered_records: List[Dict[str, Any]] = []
                for row in reader:
                    if row["customer_id"] == str(customer_id):
                        # Parse date and add to filtered records
                        row_date: datetime = datetime.strptime(row["date"], "%d/%m/%Y")
                        filtered_records.append(
                            {
                                "date": row_date.date(),
                                "amount": float(row["amount"]),
                                "source": row["source"],
                            }
                        )

                if not filtered_records:
                    dispatcher.utter_message(
                        f"No transactions found for customer {customer_id} on "
                        f"{bill_date_obj.date().strftime('%B %Y')}."
                    )
                    return []

                # Format the output
                response1 = "Here is a summary of your costs :"
                dispatcher.utter_message(response1)
                response = "\n".join(
                    [
                        (
                            f"{record['date']} | {record['amount']} $ "
                            f"| {record['source']}"
                        )
                        for record in filtered_records
                    ]
                )

                # Send response to user
                dispatcher.utter_message(response)
                return []

        except FileNotFoundError:
            dispatcher.utter_message("Billing data file not found.")
            return []
        except Exception:
            dispatcher.utter_message("Error retrieving billing information.")
            return []
