"""
This module regroups custom validation functions, and it is
loaded as an extension of the pykwalify library:

https://pykwalify.readthedocs.io/en/latest/extensions.html#extensions
"""
from typing import Any, List, Dict, Text

from pykwalify.errors import SchemaError


def require_response_keys(
    responses: List[Dict[Text, Any]], rule_obj: Dict, path: Text
) -> bool:
    """
    Validate that response dicts have either the "text" key or the "custom" key.
    """
    for response in responses:
        if not isinstance(response, dict):
            # this is handled by other validation rules
            continue

        if response.get("text") is None and not response.get("custom"):
            raise SchemaError("Missing 'text' or 'custom' key in response.")

    return True
