"""This module regroups custom validation functions, and it is
loaded as an extension of the pykwalify library:

https://pykwalify.readthedocs.io/en/latest/extensions.html#extensions
"""

from typing import Any, Dict, List, Text, Union

from pykwalify.errors import SchemaError


def require_response_keys(
    responses: List[Dict[Text, Any]], _: Dict, __: Text
) -> Union[SchemaError, bool]:
    """Validates that response dicts have either the "text" key or the "custom" key."""
    for response in responses:
        if not isinstance(response, dict):
            # this is handled by other validation rules
            continue

        if response.get("text") is None and not response.get("custom"):
            return SchemaError(
                "Missing 'text' or 'custom' key in response or "
                "null 'text' value in response."
            )

        conditions = response.get("condition", [])
        if isinstance(conditions, str):
            continue

        for condition in conditions:
            if not isinstance(condition, dict):
                return SchemaError("Condition must be a dictionary.")
            if not all(key in condition for key in ("type", "name", "value")):
                return SchemaError(
                    "Condition must have 'type', 'name', and 'value' keys."
                )

            if condition.get("type") != "slot":
                return SchemaError("Condition type must be of type `slot`.")

    return True
