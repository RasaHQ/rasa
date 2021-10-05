from pathlib import Path
from typing import Union

import rasa.shared.utils.io
from rasa.shared.constants import REQUIRED_SLOTS_KEY


def migrate_domain_format(domain_file: Union[list, Path], out_file: Path) -> None:
    """Converts 2.0 domain to 3.0 format."""
    if isinstance(domain_file, list):
        domain_file = domain_file[0]

    original_content = rasa.shared.utils.io.read_yaml_file(domain_file)

    intents = original_content.get("intents", [])
    entities = original_content.get("entities", [])
    slots = original_content.get("slots", {})
    forms = original_content.get("forms", {})
    responses = original_content.get("responses", {})
    actions = original_content.get("actions", [])
    e2e_actions = original_content.get("e2e_actions", [])
    session_config = original_content.get("session_config", {})

    new_slots = {}
    new_forms = {}

    for form_name, form_data in forms.items():
        if form_data is not None and REQUIRED_SLOTS_KEY not in form_data:
            forms[form_name] = {REQUIRED_SLOTS_KEY: form_data}

        required_slots = []
        for slot_name, mappings in form_data.items():
            slot_properties = slots.get(slot_name)
            slot_properties.update({"mappings": mappings})
            slots[slot_name] = slot_properties
            required_slots.append(slot_name)
        new_forms[form_name] = {REQUIRED_SLOTS_KEY: required_slots}

    for slot_name, properties in slots.items():
        if slot_name in entities:
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
        "version": "3.0",
        "intents": intents,
        "entities": entities,
        "slots": new_slots,
        "responses": responses,
        "actions": actions,
        "e2e_actions": e2e_actions,
        "forms": new_forms,
        "session_config": session_config,
    }

    rasa.shared.utils.io.write_yaml(new_domain, out_file, True)
