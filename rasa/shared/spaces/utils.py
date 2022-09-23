from typing import Dict, Text

PREFIXING_SEPARATOR = "!"
UTTER_PREFIX = "utter_"


def prefix_string(s: Text, prefix: Text) -> Text:
    return f"{prefix}{PREFIXING_SEPARATOR}{s}"


def prefix_dict_value(d: Dict, key: Text, prefix: Text) -> None:
    """Prefix a value in a dictionary."""
    if key in d:
        d[key] = f"{prefix}{PREFIXING_SEPARATOR}{d[key]}"
    else:
        raise ValueError(f"Key {key} is not present in dictionary {d}")


def prefix_dict_value_with_potential_utter(d: Dict, key: Text, prefix: Text) -> None:
    """Prefix a value in a dictionary and treat utter_ actions separately."""
    if key in d:
        if d[key].startswith(UTTER_PREFIX):
            d[key] = f"{UTTER_PREFIX}{prefix}{PREFIXING_SEPARATOR}" \
                     f"{d[key][len(UTTER_PREFIX):]}"
        else:
            d[key] = f"{prefix}{PREFIXING_SEPARATOR}{d[key]}"
    else:
        raise ValueError(f"Key {key} is not present in dictionary {d}")


def prefix_dict_key(d: Dict, key: Text, prefix: Text) -> None:
    """Prefix key and delete old key."""
    if key in d:
        d[f"{prefix}{PREFIXING_SEPARATOR}{key}"] = d[key]
        del d[key]
    else:
        raise ValueError(f"Key {key} is not present in dictionary {d}")

