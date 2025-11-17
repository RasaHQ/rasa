"""This module regroups custom validation functions, and it is
loaded as an extension of the pykwalify library:

https://pykwalify.readthedocs.io/en/latest/extensions.html#extensions
"""

from typing import Any, Dict, List, Optional, Union

from pykwalify.errors import SchemaError

BOT_UTTERED_KEY = "bot_uttered"
BUTTONS_KEY = "buttons"
FLOW_STARTED_KEY = "flow_started"
PATTERN_CLARIFICATION_CONTAINS_KEY = "pattern_clarification_contains"


def run_common_dict_assertion_validations(
    assertion_dict: Dict[str, Any], assertion_key: str
) -> Optional[SchemaError]:
    if not assertion_dict:
        return SchemaError(
            f"The '{assertion_key}' assertion cannot be an empty dictionary."
        )

    operator = assertion_dict.get("operator")
    if operator is None:
        return SchemaError(
            f"The 'operator' key is missing in the '{assertion_key}' assertion."
        )
    if operator.lower() not in {"all", "any"}:
        return SchemaError(
            f"The 'operator' in the '{assertion_key}' assertion "
            f"must be either 'all' or 'any' (case insensitive)."
        )

    flow_ids = assertion_dict.get("flow_ids")

    if flow_ids is None:
        return SchemaError(
            f"The 'flow_ids' key is missing in the '{assertion_key}' assertion."
        )

    if not isinstance(flow_ids, list):
        return SchemaError(
            f"The 'flow_ids' in the '{assertion_key}' assertion must be a list."
        )

    if len(flow_ids) == 0:
        return SchemaError(
            f"The 'flow_ids' in the '{assertion_key}' assertion must "
            f"be a non-empty list."
        )

    return None


def require_assertion_keys(
    assertions: List[Dict[str, Any]], _: Dict, __: str
) -> Union[SchemaError, bool]:
    """Validates that certain assertion keys are not mapped to empty values."""
    for assertion in assertions:
        if not isinstance(assertion, dict):
            # this is handled by other validation rules
            continue

        bot_uttered_dict = assertion.get(BOT_UTTERED_KEY)
        if BOT_UTTERED_KEY in assertion and isinstance(bot_uttered_dict, dict):
            if not bot_uttered_dict:
                return SchemaError(
                    f"The '{BOT_UTTERED_KEY}' assertion is an empty dictionary."
                )

            if BUTTONS_KEY in bot_uttered_dict and not bot_uttered_dict.get(
                BUTTONS_KEY
            ):
                return SchemaError(
                    f"The '{BUTTONS_KEY}' key in the '{BOT_UTTERED_KEY}' assertion "
                    f"is mapped to a null value or empty list."
                )

        flow_started_assertion = assertion.get(FLOW_STARTED_KEY)
        if flow_started_assertion is not None:
            if not isinstance(flow_started_assertion, str) and not isinstance(
                flow_started_assertion, dict
            ):
                return SchemaError(
                    f"The '{FLOW_STARTED_KEY}' assertion must be either a string "
                    f"or a dictionary."
                )

            if (
                isinstance(flow_started_assertion, str)
                and not flow_started_assertion.strip()
            ):
                return SchemaError(
                    f"The '{FLOW_STARTED_KEY}' assertion cannot be an empty string."
                )

            if isinstance(flow_started_assertion, dict):
                result = run_common_dict_assertion_validations(
                    flow_started_assertion, FLOW_STARTED_KEY
                )
                if result is not None:
                    return result

                operator = flow_started_assertion.get("operator")
                flow_ids = flow_started_assertion.get("flow_ids", [])
                if operator and operator.lower() == "all" and len(flow_ids) != 1:
                    return SchemaError(
                        f"When using the 'all' operator in the '{FLOW_STARTED_KEY}' "
                        f"assertion, exactly one flow ID must be provided."
                    )

        pattern_clarification_contains = assertion.get(
            PATTERN_CLARIFICATION_CONTAINS_KEY
        )
        if pattern_clarification_contains is not None:
            if not isinstance(pattern_clarification_contains, list) and not isinstance(
                pattern_clarification_contains, dict
            ):
                return SchemaError(
                    f"The '{PATTERN_CLARIFICATION_CONTAINS_KEY}' assertion must be "
                    f"either a list or a dictionary."
                )

            if isinstance(pattern_clarification_contains, list):
                if not pattern_clarification_contains:
                    return SchemaError(
                        f"The '{PATTERN_CLARIFICATION_CONTAINS_KEY}' assertion "
                        f"cannot be an empty list."
                    )

                if not all(
                    isinstance(item, str) and item.strip()
                    for item in pattern_clarification_contains
                ):
                    return SchemaError(
                        f"All items in the '{PATTERN_CLARIFICATION_CONTAINS_KEY}' "
                        f"list must be non-empty strings."
                    )

            if isinstance(pattern_clarification_contains, dict):
                result = run_common_dict_assertion_validations(
                    pattern_clarification_contains, PATTERN_CLARIFICATION_CONTAINS_KEY
                )
                if result is not None:
                    return result

    return True
