import re


def interpolate_text(text, values):
    if isinstance(text, str):
        # transforming template tags from
        # "{tag_name}" to "{0[tag_name]}"
        # as described here:
        # https://stackoverflow.com/questions/7934620/python-dots-in-the-name-of-variable-in-a-format-string#comment9695339_7934969
        # assuming that slot_name do not contain newline character here
        text = re.sub(r"{([^\n]+?)}", r"{0[\1]}", text)
        return text.format(values)
    else:
        return text


def interpolate(dictionary, values):
    for k, v in dictionary.items():
        if isinstance(v, dict):
            interpolate(v, values)
        else:
            dictionary[k] = interpolate_text(v, values)
    return dictionary
