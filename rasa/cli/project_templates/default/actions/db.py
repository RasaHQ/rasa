import os
import shutil
import tempfile
from typing import Any, List

from pydantic import BaseModel

from rasa.nlu.utils import write_json_to_file
from rasa.shared.utils.io import read_json_file

ORIGIN_DB_PATH = "db"
CONTACTS = "contacts.json"


class Contact(BaseModel):
    name: str
    handle: str


def get_session_db_path(session_id: str) -> str:
    tempdir = tempfile.gettempdir()
    project_name = "calm_starter"
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


def add_contact(session_id: str, contact: Contact) -> None:
    contacts = get_contacts(session_id)
    contacts.append(contact)
    write_db(session_id, CONTACTS, [c.dict() for c in contacts])


def write_contacts(session_id: str, contacts: List[Contact]) -> None:
    write_db(session_id, CONTACTS, [c.dict() for c in contacts])
