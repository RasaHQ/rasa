import copy
import re
import logging
from jinja2 import Template
import jinja2
import structlog
from typing import Text, Dict, Union, Any, List

from rasa.core.constants import JINJA2_TEMPLATE_ENGINE, RASA_FORMAT_TEMPLATE_ENGINE

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


def _get_variables_to_be_rendered(
    response: Text, values: Dict[Text, Text]
) -> Dict[Text, Text]:
    """Get the variables that need to be rendered in the response.

    Args:
        response: The response that should be interpolated.
        values: A dictionary of keys and the values that those
            keys should be replaced with.

    Returns:
        The variables that need to be rendered.
    """
    # The regex matches and captures all the strings that are enclosed in curly braces.
    # The strings should not contain newlines or curly braces.
    variables = re.findall(r"{([^\n{}]+?)}", response)
    return {var: values[var] for var in variables if var in values}


def interpolate_format_template(response: Text, values: Dict[Text, Text]) -> Text:
    """Interpolate values into responses with placeholders.

    Transform response tags from "{tag_name}" to "{0[tag_name]}" as described here:
    https://stackoverflow.com/questions/7934620/python-dots-in-the-name-of-variable-in-a-format-string#comment9695339_7934969
    Block characters, making sure not to allow:
    (a) newline in slot name
    (b) { or } in slot name

    Args:
        response: The piece of text that should be interpolated.
        values: A dictionary of keys and the values that those
            keys should be replaced with.

    Returns:
        The piece of text with any replacements made.
    """
    try:
        values_to_be_rendered = _get_variables_to_be_rendered(response, values)
        text = re.sub(r"{([^\n{}]+?)}", r"{0[\1]}", response)
        text = text.format(values_to_be_rendered)
        if "0[" in text:
            # regex replaced tag but format did not replace
            # likely cause would be that tag name was enclosed
            # in double curly and format func simply escaped it.
            # we don't want to return {0[SLOTNAME]} thus
            # restoring original value with { being escaped.
            return response.format({})

        return text
    except KeyError as e:
        event_info = (
            "The specified slot name does not exist, "
            "and no explicit value was provided during the response invocation. "
            "Return the response without populating it."
        )
        structlogger.exception(
            "interpolator.interpolate.text",
            response=copy.deepcopy(response),
            placeholder_key=e.args[0],
            event_info=event_info,
        )
        return response


def interpolate_jinja_template(response: Text, values: Dict[Text, Any]) -> Text:
    """Interpolate values into responses with placeholders using jinja.

    Args:
        response: The piece of text that should be interpolated.
        values: A dictionary of keys and the values that those
            keys should be replaced with.

    Returns:
    The piece of text with any replacements made.
    """
    try:
        return Template(response).render(values)
    except jinja2.exceptions.UndefinedError as e:
        event_info = (
            "The specified slot name does not exist, "
            "and no explicit value was provided during the response invocation. "
            "Return the response without populating it."
        )
        structlogger.exception(
            "interpolator.interpolate.text",
            response=copy.deepcopy(response),
            placeholder_key=e.args[0],
            event_info=event_info,
        )
        return response


def interpolate(
    response: Union[List[Any], Dict[Text, Any], Text],
    values: Dict[Text, Text],
    method: str,
) -> Union[List[Any], Dict[Text, Any], Text]:
    """Recursively process response and interpolate any text keys.

    Args:
        response: The response that should be interpolated.
        values: A dictionary of keys and the values that those
            keys should be replaced with.
        method: The method to use for interpolation. If `None` or `"format"`,

    Returns:
        The response with any replacements made.
    """
    if method == RASA_FORMAT_TEMPLATE_ENGINE:
        interpolator = interpolate_format_template
    elif method == JINJA2_TEMPLATE_ENGINE:
        interpolator = interpolate_jinja_template
    else:
        raise ValueError(f"Unknown interpolator implementation '{method}'")

    if isinstance(response, str):
        return interpolator(response, values)
    elif isinstance(response, dict):
        for k, v in response.items():
            if isinstance(v, dict):
                interpolate(v, values, method)
            elif isinstance(v, list):
                response[k] = [interpolate(i, values, method) for i in v]
            elif isinstance(v, str):
                response[k] = interpolator(v, values)
        return response
    elif isinstance(response, list):
        return [interpolate(i, values, method) for i in response]
    return response
