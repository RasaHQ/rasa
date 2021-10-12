from pathlib import Path

import rasa.shared.utils.io
from rasa.shared.constants import REQUIRED_SLOTS_KEY, IGNORED_INTENTS
from rasa.shared.core.domain import (
    KEY_INTENTS,
    KEY_ENTITIES,
    KEY_SLOTS,
    KEY_FORMS,
    KEY_RESPONSES,
    KEY_ACTIONS,
    KEY_E2E_ACTIONS,
    SESSION_CONFIG_KEY,
)


def migrate_domain_format(domain_file: Path, out_file: Path) -> None:
    """Converts 2.0 domain to 3.0 format."""
    current_dir = domain_file.parent

    if domain_file.is_dir():
        original_content = {}
        backup = current_dir / "original_domain"
        backup.mkdir()
        for file in domain_file.iterdir():
            file_content = rasa.shared.utils.io.read_yaml_file(file)
            original_content.update(file_content)
            rasa.shared.utils.io.write_yaml(file_content, backup / file.name, True)
    else:
        original_content = rasa.shared.utils.io.read_yaml_file(domain_file)
        backup = current_dir / "original_domain.yml"
        rasa.shared.utils.io.write_yaml(original_content, backup, True)

    intents = original_content.get(KEY_INTENTS, [])
    entities = original_content.get(KEY_ENTITIES, [])
    slots = original_content.get(KEY_SLOTS, {})
    forms = original_content.get(KEY_FORMS, {})
    responses = original_content.get(KEY_RESPONSES, {})
    actions = original_content.get(KEY_ACTIONS, [])
    e2e_actions = original_content.get(KEY_E2E_ACTIONS, [])
    session_config = original_content.get(SESSION_CONFIG_KEY, {})

    new_slots = {}
    new_forms = {}

    for form_name, form_data in forms.items():
        ignored_intents = form_data.pop(IGNORED_INTENTS, [])
        if REQUIRED_SLOTS_KEY in form_data:
            form_data = form_data.get(REQUIRED_SLOTS_KEY, {})

        required_slots = []
        for slot_name, mappings in form_data.items():
            slot_properties = slots.get(slot_name)
            slot_properties.update({"mappings": mappings})
            slots[slot_name] = slot_properties
            required_slots.append(slot_name)
        new_forms[form_name] = {
            IGNORED_INTENTS: ignored_intents,
            REQUIRED_SLOTS_KEY: required_slots,
        }

    for slot_name, properties in slots.items():
        if slot_name in entities and properties.get("auto_fill", True) is True:
            from_entity_mapping = {
                "type": "from_entity",
                "entity": slot_name,
            }
            mappings = properties.get("mappings", [])
            if from_entity_mapping not in mappings:
                mappings.append(from_entity_mapping)
                properties.update({"mappings": mappings})

        if "auto_fill" in properties:
            del properties["auto_fill"]

        new_slots[slot_name] = properties

    new_domain = {
        "version": "2.0",
        KEY_INTENTS: intents,
        KEY_ENTITIES: entities,
        KEY_SLOTS: new_slots,
        KEY_RESPONSES: responses,
        KEY_ACTIONS: actions,
        KEY_E2E_ACTIONS: e2e_actions,
        KEY_FORMS: new_forms,
        SESSION_CONFIG_KEY: session_config,
    }

    rasa.shared.utils.io.write_yaml(new_domain, out_file, True)
