import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Database:
    def __init__(self, csv_path: Optional[Path] = None) -> None:
        """Initialize the database with CSV file paths."""
        self.project_root_path = Path(__file__).resolve().parent.parent
        self.csv_path = csv_path or self.project_root_path / "csvs"
        self.logger = self.setup_logger()

    def setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)8s %(name)s  - %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _read_csv(self, filename: str) -> List[Dict[str, Any]]:
        """Read CSV file and return list of dictionaries."""
        try:
            with open(self.csv_path / filename, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                return list(reader)
        except Exception as e:
            self.logger.error(f"Error reading CSV file {filename}: {e}")
            return []

    def _write_csv(self, filename: str, data: List[Dict[str, Any]]) -> bool:
        """Write list of dictionaries to CSV file."""
        try:
            if not data:
                return True

            with open(self.csv_path / filename, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return True
        except Exception as e:
            self.logger.error(f"Error writing CSV file {filename}: {e}")
            return False

    def get_user_by_name(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information by username."""
        try:
            users = self._read_csv("users.csv")
            for user in users:
                if user["name"] == username:
                    return user
            return None
        except Exception as e:
            self.logger.error(f"Error getting user by name: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information by user_id."""
        try:
            users = self._read_csv("users.csv")
            for user in users:
                if int(user["id"]) == user_id:
                    return user
            return None
        except Exception as e:
            self.logger.error(f"Error getting user by id: {e}")
            return None

    def get_account_by_user_and_number(
        self, user_id: int, account_number: str
    ) -> Optional[Dict[str, Any]]:
        """Get account information by user_id and account number."""
        try:
            accounts = self._read_csv("accounts.csv")
            for account in accounts:
                if (
                    int(account["user_id"]) == user_id
                    and account["number"] == account_number
                ):
                    return account
            return None
        except Exception as e:
            self.logger.error(f"Error getting account: {e}")
            return None

    def get_accounts_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all accounts for a user."""
        try:
            accounts = self._read_csv("accounts.csv")
            return [
                account for account in accounts if int(account["user_id"]) == user_id
            ]

        except Exception as e:
            self.logger.error(f"Error getting accounts by user: {e}")
            return []

    def get_payees_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all payees for a user."""
        try:
            payees = self._read_csv("payees.csv")
            return [payee for payee in payees if int(payee["user_id"]) == user_id]
        except Exception as e:
            self.logger.error(f"Error getting payees by user: {e}")
            return []

    def get_payee_by_name_and_user(
        self, payee_name: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get payee information by name and user_id."""
        try:
            payees = self._read_csv("payees.csv")
            for payee in payees:
                if payee["name"] == payee_name and int(payee["user_id"]) == user_id:
                    return payee
            return None
        except Exception as e:
            self.logger.error(f"Error getting payee by name and user: {e}")
            return None

    def get_cards_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all cards for a user."""
        try:
            cards = self._read_csv("cards.csv")
            return [card for card in cards if int(card["user_id"]) == user_id]
        except Exception as e:
            self.logger.error(f"Error getting cards by user: {e}")
            return []

    def get_card_by_number(self, card_number: str) -> Optional[Dict[str, Any]]:
        """Get card information by card number."""
        try:
            cards = self._read_csv("cards.csv")
            for card in cards:
                if card["number"] == card_number:
                    return card
            return None
        except Exception as e:
            self.logger.error(f"Error getting card by number: {e}")
            return None

    def update_card_status(self, card_number: str, status: str) -> bool:
        """Update card status."""
        try:
            cards = self._read_csv("cards.csv")
            for card in cards:
                if card["number"] == card_number:
                    card["status"] = status
                    break
            return self._write_csv("cards.csv", cards)
        except Exception as e:
            self.logger.error(f"Error updating card status: {e}")
            return False

    def add_payee(
        self,
        user_id: int,
        name: str,
        sort_code: str,
        account_number: str,
        payee_type: str,
        reference: str = "",
    ) -> bool:
        """Add a new payee."""
        try:
            payees = self._read_csv("payees.csv")

            # Get the next ID
            next_id = 1
            if payees:
                next_id = max(int(payee["id"]) for payee in payees) + 1

            # Create new payee record
            new_payee = {
                "id": str(next_id),
                "user_id": str(user_id),
                "name": name,
                "sort_code": sort_code,
                "account_number": account_number,
                "type": payee_type,
                "reference": reference,
                "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add to list and save
            payees.append(new_payee)
            return self._write_csv("payees.csv", payees)
        except Exception as e:
            self.logger.error(f"Error adding payee: {e}")
            return False

    def remove_payee(self, payee_name: str, user_id: int) -> bool:
        """Remove a payee."""
        try:
            payees = self._read_csv("payees.csv")
            payees = [
                payee
                for payee in payees
                if not (
                    payee["name"] == payee_name and int(payee["user_id"]) == user_id
                )
            ]
            return self._write_csv("payees.csv", payees)
        except Exception as e:
            self.logger.error(f"Error removing payee: {e}")
            return False

    def check_sufficient_funds(
        self, user_id: int, account_number: str, amount: float
    ) -> bool:
        """Check if account has sufficient funds."""
        try:
            account = self.get_account_by_user_and_number(user_id, account_number)
            if not account:
                return False
            return float(account["balance"]) >= amount
        except Exception as e:
            self.logger.error(f"Error checking sufficient funds: {e}")
            return False

    def get_branches(self) -> List[Dict[str, Any]]:
        """Get all branches."""
        try:
            return self._read_csv("branches.csv")
        except Exception as e:
            self.logger.error(f"Error getting branches: {e}")
            return []

    def get_advisors_by_branch(self, branch_id: int) -> List[Dict[str, Any]]:
        """Get all advisors for a branch."""
        try:
            advisors = self._read_csv("advisors.csv")
            return [
                advisor
                for advisor in advisors
                if int(advisor["branch_id"]) == branch_id
            ]
        except Exception as e:
            self.logger.error(f"Error getting advisors by branch: {e}")
            return []

    def get_appointments_by_advisor(self, advisor_id: int) -> List[Dict[str, Any]]:
        """Get all appointments for an advisor."""
        try:
            appointments = self._read_csv("appointments.csv")
            return [
                appointment
                for appointment in appointments
                if int(appointment["advisor_id"]) == advisor_id
            ]
        except Exception as e:
            self.logger.error(f"Error getting appointments by advisor: {e}")
            return []

    def __enter__(self) -> "Database":
        """Enter the runtime context related to this object."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """Exit the runtime context related to this object."""
        self.logger.info("Database connection closed")
