from typing import TYPE_CHECKING, Any, Dict, Set, Text

from rasa.shared.utils.io import raise_deprecation_warning

if TYPE_CHECKING:
    from rasa.shared.core.flows.flow import FlowLanguageTranslation


RESET_PROPERTY_NAME = "reset_after_flow_ends"
PERSIST_PROPERTY_NAME = "persisted_slots"


def warn_deprecated_collect_step_config() -> None:
    """Warns about deprecated reset_after_flow_ends usage in collect steps."""
    raise_deprecation_warning(
        f"Configuring '{RESET_PROPERTY_NAME}' in collect steps is "
        f"deprecated and will be removed in Rasa Pro 4.0.0. In the parent flow, "
        f"please use the '{PERSIST_PROPERTY_NAME}' "
        "property at the flow level instead."
    )


def get_duplicate_slot_persistence_config_error_message(
    flow_id: str, collect_step: str
) -> str:
    """Returns an error message for duplicate slot persistence configuration."""
    return (
        f"Flow with id '{flow_id}' uses the '{RESET_PROPERTY_NAME}' property "
        f"in collect step '{collect_step}' and also the "
        f"'{PERSIST_PROPERTY_NAME}' property at the flow level. "
        "Please use only one of the two configuration methods."
    )


def get_invalid_slot_persistence_config_error_message(
    flow_id: str, invalid_slots: Set[str]
) -> str:
    """Returns an error message for invalid slot persistence configuration."""
    return (
        f"Flow with id '{flow_id}' lists slot(s) '{invalid_slots}' in the "
        f"'{PERSIST_PROPERTY_NAME}' property. However these slots "
        f"are neither used in a collect step nor a set_slot step of the flow. "
        f"Please remove such slots from the '{PERSIST_PROPERTY_NAME}' property."
    )


def extract_translations(
    translation_data: Dict[Text, Any],
) -> Dict[Text, "FlowLanguageTranslation"]:
    """Extracts translations from a dictionary.

    Args:
        translation_data: The dictionary containing the translations.

    Returns:
        A dictionary containing the extracted translations.
    """
    from rasa.shared.core.flows.flow import FlowLanguageTranslation

    return {
        language_code: FlowLanguageTranslation.parse_obj({**data})
        for language_code, data in translation_data.items()
        if language_code != "metadata"
    }
