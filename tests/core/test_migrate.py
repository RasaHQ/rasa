import itertools
import shutil
import re
import textwrap
from pathlib import Path
from typing import Text, Any, Tuple, Callable
from unittest.mock import patch, call

import pytest

import rasa.shared.utils.io
from rasa.core import migrate
from rasa.shared.core.domain import Domain
from rasa.shared.exceptions import RasaException
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION


def prepare_domain_path(directory: Path, domain_content: Text, file_name: Text) -> Path:
    original_content = textwrap.dedent(domain_content)
    domain_file = directory / file_name
    rasa.shared.utils.io.write_text_file(original_content, domain_file)
    return domain_file


@pytest.fixture()
def domain_out_file(tmp_path: Path) -> Path:
    return tmp_path / "custom_new_domain.yml"


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

    migrate.migrate_domain_format(existing_domain_file, domain_out_file)

    domain = Domain.from_path(domain_out_file)
    assert domain

    old_domain_path = tmp_path / "original_domain.yml"
    assert old_domain_path

    migrated_domain = rasa.shared.utils.io.read_yaml_file(domain_out_file)

    migrated_training_data_version = migrated_domain.get("version")
    assert migrated_training_data_version == LATEST_TRAINING_DATA_FORMAT_VERSION

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
                        {"active_loop": "booking_form", "requested_slot": "email"}
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
    migrate.migrate_domain_format(existing_domain_file, domain_out_file)

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
                        {"active_loop": "booking_form", "requested_slot": "email"}
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
    migrate.migrate_domain_format(existing_domain_file, domain_out_file)
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
                        }
                    ],
                }
            ],
        }
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

    migrate.migrate_domain_format(domain_dir, domain_out_dir)
    domain = Domain.from_directory(str(domain_out_dir))

    assert domain

    old_domain_path = tmp_path / "original_domain"
    assert old_domain_path.exists()

    for file in old_domain_path.iterdir():
        assert file.name in ["slots.yml", "forms.yml"]

    for file in domain_out_dir.iterdir():
        assert file.name in ["slots.yml", "forms.yml"]
        migrated_file = rasa.shared.utils.io.read_yaml_file(file)

        migrated_training_data_version = migrated_file.get("version")
        assert migrated_training_data_version == LATEST_TRAINING_DATA_FORMAT_VERSION


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
    migrate.migrate_domain_format(existing_domain_file, domain_out_file)
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

    migrated_training_data_version = migrated_domain.get("version")
    assert migrated_training_data_version == LATEST_TRAINING_DATA_FORMAT_VERSION


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
        migrate.migrate_domain_format(existing_domain_file, domain_out_file)

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


def test_migrate_domain_with_no_requested_slot_for_from_entity_mappings(
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
          location:
            type: text
            influence_conversation: false
          email:
            type: text
            influence_conversation: false
        forms:
            some_form:
                location:
                - entity: city
                  type: from_entity
                - intent: something
                  type: from_text
        """,
        "domain.yml",
    )

    migrate.migrate_domain_format(existing_domain_file, domain_out_file)

    domain = Domain.from_path(domain_out_file)
    assert domain

    migrated_domain = rasa.shared.utils.io.read_yaml_file(domain_out_file)
    migrated_slots = migrated_domain.get("slots")
    location_slot = migrated_slots.get("location")
    mappings = location_slot.get("mappings")
    assert mappings[0] == {
        "entity": "city",
        "type": "from_entity",
        "conditions": [{"active_loop": "some_form"}],
    }
    assert mappings[1] == {
        "intent": "something",
        "type": "from_text",
        "conditions": [{"active_loop": "some_form", "requested_slot": "location"}],
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
    migrate.migrate_domain_format(existing_domain_file, domain_out_file)

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
                    {"active_loop": "form_one", "requested_slot": "location"}
                ],
            },
            {
                "type": "from_entity",
                "entity": "city",
                "conditions": [{"active_loop": "form_two"}],
            },
        ],
    }


def test_migrate_domain_dir_with_out_path_None(tmp_path: Path):
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

    migrate.migrate_domain_format(domain_dir, None)

    domain_out = tmp_path / "new_domain"
    assert domain_out.exists()

    for file in domain_out.iterdir():
        assert file.name in ["slots.yml", "forms.yml"]

    domain = Domain.from_directory(str(domain_out))
    assert domain


def test_migrate_domain_multiple_files_with_duplicate_slots(tmp_path: Path):
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
        migrate.migrate_domain_format(domain_dir, None)


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
        migrate.migrate_domain_format(domain_dir, None)


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

    new_domain_dir = tmp_path / "migrated_domain"
    migrate.migrate_domain_format(domain_dir, new_domain_dir)
    domain = Domain.from_directory(new_domain_dir)
    assert domain

    for file in new_domain_dir.iterdir():
        migrated = rasa.shared.utils.io.read_yaml_file(file)

        migrated_training_data_version = migrated.get("version")
        assert migrated_training_data_version == LATEST_TRAINING_DATA_FORMAT_VERSION

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

    new_domain_file = tmp_path / "new_domain.yml"

    with pytest.raises(
        RasaException,
        match=f"The file '{domain_file.as_posix()}' could not "
        f"be validated as a domain file.",
    ):
        migrate.migrate_domain_format(domain_file, new_domain_file)


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
        match=f"The domain directory '{domain_dir.as_posix()}' does not contain any "
        f"domain files. Please make sure to include these for a successful "
        f"migration.",
    ):
        migrate.migrate_domain_format(domain_dir, None)


def test_migrate_domain_raises_for_missing_slots_and_forms(tmp_path: Path):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        entities:
        - bla
        """,
        "domain.yml",
    )

    with pytest.raises(
        RasaException,
        match=f"The files you have provided in '{re.escape(str(domain_dir))}' "
        f"are missing slots or forms. "
        f"Please make sure to include these for a successful migration.",
    ):
        migrate.migrate_domain_format(domain_dir, None)


def test_migrate_domain_raises_when_migrated_files_are_found(tmp_path: Path):
    domain_dir = tmp_path / "domain"
    domain_dir.mkdir()
    prepare_domain_path(
        domain_dir,
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents: []
        """,
        "domain.yml",
    )

    with pytest.raises(
        RasaException, match="Some of the given files (.*) have already been migrated.*"
    ):
        migrate.migrate_domain_format(domain_dir, None)


def test_migrate_folder_only_migrates_domain_files(tmp_path: Path):
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

    prepare_domain_path(
        domain_dir,
        """
        not a domain file.
        """,
        "not-a-domain-file.yml",
    )

    out_dir = tmp_path / "out_dir"
    migrate.migrate_domain_format(domain_dir, out_dir)
    assert set(f.name for f in out_dir.iterdir()) == {"forms.yml", "slots.yml"}
    # i.e. the not-a-domain-file is not migrated


def example_migrate_folder_fails_because_multiple_slots_sections(
    path: Path,
) -> Tuple[Path, Text]:
    domain_dir = path / "domain"
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

    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        responses:
          utter_greet:
          - text: "Hi there!"
        """,
        "responses.yml",
    )

    return domain_dir, "Domain files with multiple 'slots' sections were " "provided."


@pytest.mark.parametrize(
    "test_case, default_out_dir, out_dir_exists",
    [
        (test_case, default_out_dir, out_dir_exists)
        for test_case in [example_migrate_folder_fails_because_multiple_slots_sections]
        for (default_out_dir, out_dir_exists) in [
            (True, True),
            (True, False),
            (False, False),
        ]
    ],
)
def test_migrate_domain_cleanups_after_raising(
    tmp_path: Path,
    test_case: Callable[[Path], Tuple[Text, Text]],
    default_out_dir: bool,
    out_dir_exists: bool,
):
    # input
    out_path = None if default_out_dir else (tmp_path / "custom_out_path")
    domain_path, error_msg_match = test_case(tmp_path)
    migrating_file_only = domain_path.is_file()
    domain_files = list(domain_path.iterdir())

    # paths to be used by migration tool
    domain_parent_dir = domain_path.parent
    expected_out_path = (
        out_path
        if out_path is not None
        else (domain_parent_dir / migrate.DEFAULT_NEW_DOMAIN)
    )
    expected_backup_path = domain_parent_dir / migrate.ORIGINAL_DOMAIN
    if migrating_file_only:
        expected_backup_path = f"{expected_backup_path}{migrate.YML_SUFFIX}"
        expected_out_path = f"{expected_out_path}{migrate.YML_SUFFIX}"

    # create the folder if needed
    if not migrating_file_only and out_dir_exists:
        expected_out_path.mkdir(parents=True)

    # migrate!
    with pytest.raises(RasaException, match=error_msg_match):
        migrate.migrate_domain_format(domain_path, out_path)
    assert Path.exists(domain_path)
    assert all(Path.exists(file) for file in domain_files)
    assert not Path.exists(expected_backup_path)
    if not migrating_file_only:
        # if and only if the folder didn't exist before, it should not exist afterwards
        assert Path.exists(expected_out_path) == out_dir_exists

    # ... and to assert we really did remove something:
    expected_to_be_removed = [call(expected_backup_path)]
    if not out_dir_exists:  # only removed if it didn't exist
        expected_to_be_removed.append(call(expected_out_path))
    patching = [shutil, "rmtree"] if not migrating_file_only else [Path, "unlink"]
    with patch.object(*patching) as removal:
        with pytest.raises(RasaException, match=error_msg_match):
            migrate.migrate_domain_format(domain_path, out_path)
        assert removal.call_count == 1 + (not out_dir_exists)
        removal.assert_has_calls(expected_to_be_removed)


@pytest.mark.parametrize("migrate_file_only", [True, False])
def test_migrate_domain_raises_when_backup_location_exists(
    tmp_path: Path, migrate_file_only: bool
):
    domain_dir = tmp_path / "domain"
    domain_file_name = "domain.yml"
    domain_dir.mkdir()
    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        intents: []
        """,
        domain_file_name,
    )

    if not migrate_file_only:
        domain_path = domain_dir
        backup_location = tmp_path / migrate.ORIGINAL_DOMAIN
        backup_location.mkdir()
    else:
        domain_path = domain_dir / domain_file_name
        backup_location = domain_dir / (migrate.ORIGINAL_DOMAIN + migrate.YML_SUFFIX)
        with open(backup_location, "w"):
            pass

    with pytest.raises(
        RasaException, match="The domain could not be migrated since .* exists.*"
    ):
        migrate.migrate_domain_format(domain_path, None)


@pytest.mark.parametrize(
    "migrate_file_only, default_location", itertools.combinations([True, False], 2)
)
def test_migrate_domain_raises_when_output_location_is_used(
    tmp_path: Path, migrate_file_only: bool, default_location: bool
):
    domain_dir = tmp_path / "domain"
    domain_file_name = "domain.yml"
    domain_dir.mkdir()
    prepare_domain_path(
        domain_dir,
        """
        version: "2.0"
        intents: []
        """,
        domain_file_name,
    )

    if not migrate_file_only:
        domain_path = domain_dir
        if default_location:
            out_path = None
            non_empty_existing_dir = tmp_path / migrate.DEFAULT_NEW_DOMAIN
        else:
            out_path = tmp_path / "my_custom_new_domain_name"
            non_empty_existing_dir = out_path
        non_empty_existing_dir.mkdir()
        # in contrast to the backup location, the output directory may exist
        # but must be empty
        with open(non_empty_existing_dir / "bla.txt", "w"):
            pass

    else:
        domain_path = domain_dir / domain_file_name
        if default_location:
            out_path = None
            existing_file = tmp_path / (migrate.DEFAULT_NEW_DOMAIN + migrate.YML_SUFFIX)
        else:
            out_path = tmp_path / "my_custom_file_name.yml"
            existing_file = out_path
        with open(existing_file, "w"):
            pass

    with pytest.raises(
        RasaException,
        match="The domain could not be migrated to .* because .* already exists.*",
    ):
        migrate.migrate_domain_format(domain_path, out_path)
