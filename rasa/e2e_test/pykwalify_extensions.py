"""This module regroups custom validation functions, and it is
loaded as an extension of the pykwalify library:

https://pykwalify.readthedocs.io/en/latest/extensions.html#extensions
"""

from typing import Any, Dict, List, Union

from pykwalify.errors import SchemaError

BOT_UTTERED_KEY = "bot_uttered"
BUTTONS_KEY = "buttons"


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

    return True
