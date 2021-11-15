import re
import os
import textwrap
from pathlib import Path
from typing import Text, Any

import pytest

import rasa.shared.utils.io
import rasa.core.migrate
from rasa.shared.core.domain import Domain
from rasa.shared.exceptions import RasaException


def prepare_domain_path(directory: Path, domain_content: Text, file_name: Text) -> Path:
    original_content = textwrap.dedent(domain_content)
    domain_file = directory / file_name
    rasa.shared.utils.io.write_text_file(original_content, domain_file)
    return domain_file


@pytest.fixture()
def domain_out_file(tmp_path: Path) -> Path:
    return tmp_path / "new_domain.yml"


def test_migrate_domain_format_with_required_slots(
    tmp_path: Path, domain_out_file: Path
):
    existing_domain_file = prepare_domain_path(
        tmp_path,
        """
        version: "2.0"
        intents:
        - greet
        - affirm
        - inform
        entities:
        - city
        - name
        slots:
          location:
            type: text
            influence_conversation: false
          name:
            type: text
            influence_conversation: false
            auto_fill: false
          email:
            type: text
            influence_conversation: false
        forms:
           booking_form:
               ignored_intents:
               - greet
               required_slots:
                 location:
                 - type: from_entity
                   entity: city
                 email:
                 - type: from_text
                   intent: inform
                 name:
                 - type: from_entity
                   entity: surname
        """,
        "domain.yml",
    )

    rasa.core.migrate.migrate_domain_format(existing_domain_file, domain_out_file)

    domain = Domain.from_path(domain_out_file)
    assert domain

    old_domain_path = tmp_path / "original_domain.yml"
    assert old_domain_path

    migrated_domain = rasa.shared.utils.io.read_yaml_file(domain_out_file)

    migrated_slots = migrated_domain.get("slots")
    expected_slots = {
        "location": {
            "type": "text",
            "influence_conversation": False,
            "mappings": [
                {
                    "type": "from_entity",
                    "entity": "city",
                    "conditions": [{"active_loop": "booking_form"}],
                }
            ],
        },
        "name": {
            "type": "text",
            "influence_conversation": False,
            "mappings": [
                {
                    "type": "from_entity",
                    "entity": "surname",
                    "conditions": [{"active_loop": "booking_form"}],
                }
            ],
        },
        "email": {
            "type": "text",
            "influence_conversation": False,
            "mappings": [
                {
                    "type": "from_text",
                    "intent": "inform",
                    "conditions": [
                        {"active_loop": "booking_form", "requested_slot": "email"},
                    ],
                }
            ],
        },
    }
    assert migrated_slots == expected_slots

    migrated_forms = migrated_domain.get("forms")
    expected_forms = {
        "booking_form": {
            "ignored_intents": ["greet"],
            "required_slots": ["location", "email", "name"],
        }
    }
    assert migrated_forms == expected_forms


def test_migrate_domain_form_without_required_slots(
    tmp_path: Path, domain_out_file: Path
):
    existing_domain_file = prepare_domain_path(
        tmp_path,
        """
        version: "2.0"
        intents:
        - greet
        - affirm
        - inform
        entities:
        - city
        - name
        - surname
        slots:
          location:
            type: text
            influence_conversation: false
          name:
            type: text
            influence_conversation: false
            auto_fill: false
          email:
            type: text
            influence_conversation: false
        forms:
           booking_form:
               ignored_intents:
               - greet
               location:
                 - type: from_entity
                   entity: city
               email:
                 - type: from_text
                   intent: inform
               name:
                 - type: from_entity
                   entity: surname
        """,
        "domain.yml",
    )

    rasa.core.migrate.migrate_domain_format(existing_domain_file, domain_out_file)

    domain = Domain.from_path(domain_out_file)
    assert domain

    old_domain_path = tmp_path / "original_domain.yml"
    assert old_domain_path

    migrated_domain = rasa.shared.utils.io.read_yaml_file(domain_out_file)

    migrated_slots = migrated_domain.get("slots")
    expected_slots = {
        "location": {
            "type": "text",
            "influence_conversation": False,
            "mappings": [
                {
                    "type": "from_entity",
                    "entity": "city",
                    "conditions": [{"active_loop": "booking_form"}],
                }
            ],
        },
        "name": {
            "type": "text",
            "influence_conversation": False,
            "mappings": [
                {
                    "type": "from_entity",
                    "entity": "surname",
                    "conditions": [{"active_loop": "booking_form"}],
                }
            ],
        },
        "email": {
            "type": "text",
            "influence_conversation": False,
            "mappings": [
                {
                    "type": "from_text",
                    "intent": "inform",
                    "conditions": [
                        {"active_loop": "booking_form", "requested_slot": "email"},
                    ],
                }
            ],
        },
    }
    assert migrated_slots == expected_slots

    migrated_forms = migrated_domain.get("forms")
    expected_forms = {
        "booking_form": {
            "ignored_intents": ["greet"],
            "required_slots": ["location", "email", "name"],
        }
    }
    assert migrated_forms == expected_forms


@pytest.mark.parametrize(
    "slot_type,value",
    [
        ("bool", True),
        ("float", 1),
        ("text", "out"),
        ("categorical", "test"),
        ("list", ["out"]),
        ("any", "etc"),
    ],
)
def test_migrate_domain_with_diff_slot_types(
    slot_type: Text, value: Any, tmp_path: Path, domain_out_file: Path
):
    existing_domain_file = prepare_domain_path(
        tmp_path,
        f"""
        version: "2.0"
        entities:
            - outdoor
        slots:
          outdoor_seating:
           type: {slot_type}
           influence_conversation: false
        forms:
          reservation_form:
            required_slots:
               outdoor_seating:
               - type: from_intent
                 value: {value}
                 intent: confirm
        """,
        "domain.yml",
    )
    rasa.core.migrate.migrate_domain_format(existing_domain_file, domain_out_file)
    domain = Domain.from_path(domain_out_file)
    assert domain

    migrated_domain = rasa.shared.utils.io.read_yaml_file(domain_out_file)
    migrated_slots = migrated_domain.get("slots")
    expected_slots = {
        "outdoor_seating": {
            "type": slot_type,
            "influence_conversation": False,
            "mappings": [
                {
                    "type": "from_intent",
                    "value": value,
                    "intent": "confirm",
                    "conditions": [
                        {
                            "active_loop": "reservation_form",
                            "requested_slot": "outdoor_seating",
                        },
                    ],
                }
            ],
        },
    }
    assert migrated_slots == expected_slots


def test_migrate_domain_format_from_dir(tmp_path: Path):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        entities:
            - outdoor
        slots:
          outdoor_seating:
           type: bool
           influence_conversation: false
        """,
        "slots.yml",
    )

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        forms:
          reservation_form:
            required_slots:
               outdoor_seating:
               - type: from_intent
                 value: true
                 intent: confirm
        """,
        "forms.yml",
    )

    domain_out_dir = tmp_path / "new_domain"
    domain_out_dir.mkdir()

    rasa.core.migrate.migrate_domain_format(domain_dir, domain_out_dir)
    domain = Domain.from_directory(str(domain_out_dir))
    assert domain

    old_domain_path = tmp_path / "original_domain"
    assert old_domain_path.exists()

    for file in old_domain_path.iterdir():
        assert file.name in ["slots.yml", "forms.yml"]

    for file in domain_out_dir.iterdir():
        assert file.name in ["slots.yml", "forms.yml"]


def test_migrate_domain_all_keys(tmp_path: Path, domain_out_file: Path):
    existing_domain_file = prepare_domain_path(
        tmp_path,
        """
        version: "2.0"
        intents:
        - greet
        entities:
        - city
        slots:
          city:
            type: text
            influence_conversation: false
        responses:
            utter_greet:
            - text: "Hi there!"
        actions:
        - action_check_time
        forms:
          booking_form:
            required_slots:
              city:
              - type: from_entity
                entity: city
        """,
        "domain.yml",
    )
    rasa.core.migrate.migrate_domain_format(existing_domain_file, domain_out_file)
    domain = Domain.from_path(domain_out_file)
    assert domain

    migrated_domain = rasa.shared.utils.io.read_yaml_file(domain_out_file)
    migrated_intents = migrated_domain.get("intents")
    assert "greet" in migrated_intents

    migrated_entities = migrated_domain.get("entities")
    assert "city" in migrated_entities

    migrated_responses = migrated_domain.get("responses")
    assert "utter_greet" in migrated_responses

    migrated_actions = migrated_domain.get("actions")
    assert "action_check_time" in migrated_actions


def test_migrate_domain_format_with_custom_slot(tmp_path: Path, domain_out_file: Path):
    existing_domain_file = prepare_domain_path(
        tmp_path,
        """
        version: "2.0"
        intents:
        - greet
        - affirm
        - inform
        entities:
        - city
        - name
        slots:
          location:
            type: text
            influence_conversation: false
          name:
            type: text
            influence_conversation: false
            auto_fill: false
          email:
            type: text
            influence_conversation: false
        forms:
           booking_form:
               ignored_intents:
               - greet
               required_slots:
                 location:
                 - type: from_entity
                   entity: city
                 email:
                 - type: from_text
                   intent: inform
        """,
        "domain.yml",
    )

    with pytest.warns(UserWarning, match="A custom mapping was added to slot 'name'."):
        rasa.core.migrate.migrate_domain_format(existing_domain_file, domain_out_file)

    domain = Domain.from_path(domain_out_file)
    assert domain

    migrated_domain = rasa.shared.utils.io.read_yaml_file(domain_out_file)
    migrated_slots = migrated_domain.get("slots")
    custom_slot = migrated_slots.get("name")
    assert custom_slot == {
        "type": "text",
        "influence_conversation": False,
        "mappings": [{"type": "custom"}],
    }


def test_migrate_domain_format_duplicated_slots_in_forms(
    tmp_path: Path, domain_out_file: Path
):
    existing_domain_file = prepare_domain_path(
        tmp_path,
        """
        version: "2.0"
        intents:
        - greet
        - affirm
        - inform
        entities:
        - city
        slots:
          name:
            type: text
            influence_conversation: false
          location:
             type: text
             influence_conversation: false
        forms:
           form_one:
               required_slots:
                 name:
                 - type: from_text
                   intent: inform
                 location:
                 - type: from_text
                   intent: greet
           form_two:
               required_slots:
                 name:
                 - type: from_text
                   intent: inform
                 - type: from_intent
                   intent: deny
                   value: demo
                 location:
                 - type: from_entity
                   entity: city
        """,
        "domain.yml",
    )
    rasa.core.migrate.migrate_domain_format(existing_domain_file, domain_out_file)

    domain = Domain.from_path(domain_out_file)
    assert domain

    migrated_domain = rasa.shared.utils.io.read_yaml_file(domain_out_file)
    migrated_slots = migrated_domain.get("slots")
    slot_with_duplicate_mappings = migrated_slots.get("name")
    assert slot_with_duplicate_mappings == {
        "type": "text",
        "influence_conversation": False,
        "mappings": [
            {
                "type": "from_text",
                "intent": "inform",
                "conditions": [
                    {"active_loop": "form_one", "requested_slot": "name"},
                    {"active_loop": "form_two", "requested_slot": "name"},
                ],
            },
            {
                "type": "from_intent",
                "intent": "deny",
                "value": "demo",
                "conditions": [{"active_loop": "form_two", "requested_slot": "name"}],
            },
        ],
    }
    slot_with_different_mapping_conditions = migrated_slots.get("location")
    assert slot_with_different_mapping_conditions == {
        "type": "text",
        "influence_conversation": False,
        "mappings": [
            {
                "type": "from_text",
                "intent": "greet",
                "conditions": [
                    {"active_loop": "form_one", "requested_slot": "location"},
                ],
            },
            {
                "type": "from_entity",
                "entity": "city",
                "conditions": [{"active_loop": "form_two"}],
            },
        ],
    }


def test_migrate_domain_dir_with_out_path_as_file(tmp_path: Path):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        entities:
            - outdoor
        slots:
          outdoor_seating:
           type: bool
           influence_conversation: false
        """,
        "slots.yml",
    )

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        forms:
          reservation_form:
            required_slots:
               outdoor_seating:
               - type: from_intent
                 value: true
                 intent: confirm
        """,
        "forms.yml",
    )

    domain_out = tmp_path / "domain.yml"

    rasa.core.migrate.migrate_domain_format(domain_dir, domain_out)

    domain_out = tmp_path / "new_domain"
    assert domain_out.exists()

    for file in domain_out.iterdir():
        assert file.name in ["slots.yml", "forms.yml"]

    domain = Domain.from_directory(str(domain_out))
    assert domain


def test_migrate_domain_multiple_files_with_duplicate_slots(tmp_path: Path,):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        entities:
            - outdoor
        slots:
          outdoor_seating:
           type: bool
           influence_conversation: false
        """,
        "slots_one.yml",
    )

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        entities:
            - outdoor
        slots:
          cuisine:
           type: text
           influence_conversation: false
        """,
        "slots_two.yml",
    )

    with pytest.raises(
        RasaException,
        match="Domain files with multiple 'slots' sections were provided.",
    ):
        rasa.core.migrate.migrate_domain_format(domain_dir, domain_dir)


def test_migrate_domain_with_multiple_files_with_duplicate_forms(tmp_path: Path):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        forms:
          reservation_form:
            required_slots:
               outdoor_seating:
               - type: from_intent
                 value: true
                 intent: confirm
        """,
        "forms_one.yml",
    )

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        forms:
          reservation_form:
            required_slots:
               cuisine:
               - type: from_entity
                 entity: cuisine
        """,
        "forms_two.yml",
    )

    with pytest.raises(
        RasaException,
        match="Domain files with multiple 'forms' sections were provided.",
    ):
        rasa.core.migrate.migrate_domain_format(domain_dir, domain_dir)


def test_migrate_domain_from_dir_with_other_sections(tmp_path: Path):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()
    domain_file_one = "domain_one.yml"
    domain_file_two = "domain_two.yml"

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        entities:
        - outdoor
        slots:
          outdoor_seating:
           type: bool
           influence_conversation: false
        """,
        domain_file_one,
    )

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        intents:
        - greet
        forms:
          reservation_form:
            required_slots:
               outdoor_seating:
               - type: from_intent
                 value: true
                 intent: confirm
        """,
        domain_file_two,
    )
    rasa.core.migrate.migrate_domain_format(domain_dir, domain_dir)
    domain = Domain.from_directory(str(domain_dir))
    assert domain

    for file in domain_dir.iterdir():
        migrated = rasa.shared.utils.io.read_yaml_file(file)
        if file.name == domain_file_one:
            assert migrated.get("entities") == ["outdoor"]
        elif file.name == domain_file_two:
            assert migrated.get("intents") == ["greet"]


def test_migrate_domain_raises_exception_for_non_domain_file(tmp_path: Path):
    domain_file = prepare_domain_path(
        tmp_path,
        """
        version: "2.0"
        nlu:
        - intent: greet
          examples: |
          - hey
          - hello
          - hi
          - hello there
          - good morning
          - good evening
          - moin
          - hey there
        """,
        "domain.yml",
    )

    with pytest.raises(
        RasaException,
        match=f"The file '{re.escape(str(domain_file))}' could not "
        f"be validated as a domain file.",
    ):
        rasa.core.migrate.migrate_domain_format(domain_file, domain_file)


def test_migrate_domain_raises_for_non_domain_files(tmp_path: Path):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        nlu:
        - intent: greet
          examples: |
          - hey
          - hello
          - hi
          - hello there
          - good morning
          - good evening
          - moin
          - hey there
        """,
        "domain.yml",
    )

    with pytest.raises(
        RasaException,
        match=f"The files you have provided in '{re.escape(str(domain_dir))}' "
        f"are missing slots or forms. "
        f"Please make sure to include these for a successful migration.",
    ):
        rasa.core.migrate.migrate_domain_format(domain_dir, domain_dir)


def test_migrate_twice(tmp_path: Path,):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()
    migrated_domain_dir = tmp_path / "domain2"

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        entities:
            - outdoor
        slots:
          outdoor_seating:
           type: bool
           influence_conversation: false
        """,
        "slots_one.yml",
    )

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        entities:
            - outdoor
        slots:
          cuisine:
           type: text
           influence_conversation: false
        """,
        "slots_two.yml",
    )

    with pytest.raises(
        RasaException,
        match="Domain files with multiple 'slots' sections were provided.",
    ):
        rasa.core.migrate.migrate_domain_format(domain_dir, migrated_domain_dir)

    with pytest.raises(
        RasaException,
        match="Domain files with multiple 'slots' sections were provided.",
    ):
        rasa.core.migrate.migrate_domain_format(domain_dir, migrated_domain_dir)

    current_dir = domain_dir.parent
    assert not os.path.exists(current_dir / "original_domain")
    assert not os.path.exists(current_dir / "new_domain")
