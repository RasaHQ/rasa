import tempfile
import os
import shutil
from typing import Any, List
from pydantic import BaseModel

from rasa.nlu.utils import write_json_to_file
from rasa.shared.utils.io import read_json_file

ORIGIN_DB_PATH = "db"
CONTACTS = "contacts.json"
TRANSACTIONS = "transactions.json"
MY_ACCOUNT = "my_account.json"


class MyAccount(BaseModel):
    account: str
    funds: int


class Transaction(BaseModel):
    datetime: str
    recipient: str
    sender: str
    amount: str
    description: str

    def stringify(self):
        return f"{self.amount} from {self.sender} to " \
               f"{self.recipient} at {self.datetime}"


class Contact(BaseModel):
    name: str
    handle: str


def get_session_db_path(session_id: str) -> str:
    tempdir = tempfile.gettempdir()
    project_name = "financial_demo_flows_llms"
    return os.path.join(tempdir, project_name, session_id)


def prepare_db_file(session_id: str, db: str) -> str:
    session_db_path = get_session_db_path(session_id)
    os.makedirs(session_db_path, exist_ok=True)
    destination_file = os.path.join(session_db_path, db)
    if not os.path.exists(destination_file):
        origin_file = os.path.join(ORIGIN_DB_PATH, db)
        shutil.copy(origin_file, destination_file)
    return destination_file


def read_db(session_id: str, db: str) -> Any:
    db_file = prepare_db_file(session_id, db)
    return read_json_file(db_file)


def write_db(session_id: str, db: str, data: Any) -> None:
    db_file = prepare_db_file(session_id, db)
    write_json_to_file(db_file, data)


def get_contacts(session_id: str) -> List[Contact]:
    return [Contact(**item) for item in read_db(session_id, CONTACTS)]


def get_transactions(session_id: str):
    return [Transaction(**item) for item in read_db(session_id, TRANSACTIONS)]


def get_account(session_id: str):
    return MyAccount(**read_db(session_id, MY_ACCOUNT))


def write_account(session_id: str, account: MyAccount) -> None:
    write_db(session_id, MY_ACCOUNT, account.dict())


def add_contact(session_id: str, contact: Contact) -> None:
    contacts = get_contacts(session_id)
    contacts.append(contact)
    write_db(session_id, CONTACTS, [c.dict() for c in contacts])


def add_transaction(session_id: str, transaction: Transaction) -> None:
    transactions = get_transactions(session_id)
    transactions.append(transaction)
    write_db(session_id, TRANSACTIONS, [t.dict() for t in transactions])


def write_contacts(session_id: str, contacts: List[Contact]) -> None:
    write_db(session_id, CONTACTS, [c.dict() for c in contacts])
