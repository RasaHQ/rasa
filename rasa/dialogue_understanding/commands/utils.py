from typing import TYPE_CHECKING, List, Optional, Union

import structlog

if TYPE_CHECKING:
    from rasa.dialogue_understanding.commands import StartFlowCommand
    from rasa.shared.core.flows import FlowsList

structlogger = structlog.get_logger()


def start_flow_by_name(
    flow_name: str, flows: "FlowsList"
) -> Optional["StartFlowCommand"]:
    from rasa.dialogue_understanding.commands import StartFlowCommand

    if flow_name in flows.user_flow_ids:
        return StartFlowCommand(flow=flow_name)
    else:
        structlogger.debug(
            "command_parser.start_flow_by_name.invalid_flow_id", flow=flow_name
        )
        return None


def extract_cleaned_options(options_str: str) -> List[str]:
    """Extract and clean options from a string."""
    return sorted(
        opt.strip().strip('"').strip("'")
        for opt in options_str.split(",")
        if opt.strip()
    )


def is_none_value(value: str) -> bool:
    """Check if the value is a none value."""
    if not value:
        return True
    return value in {
        "[missing information]",
        "[missing]",
        "None",
        "undefined",
        "null",
    }


def clean_extracted_value(value: str) -> str:
    """Clean up the extracted value from the llm."""
    # replace any combination of single quotes, double quotes, and spaces
    # from the beginning and end of the string
    return value.strip("'\" ")


def get_nullable_slot_value(slot_value: str) -> Union[str, None]:
    """Get the slot value or None if the value is a none value.

    Args:
        slot_value: the value to coerce

    Returns:
        The slot value or None if the value is a none value.
    """
    return slot_value if not is_none_value(slot_value) else None
