from typing import Dict, Text, List, Any, NamedTuple

from rasa.shared.constants import (
    SESSION_CONFIG_KEY,
    KEY_INTENTS,
    KEY_ENTITIES,
    KEY_FORMS,
    KEY_ACTIONS,
    KEY_E2E_ACTIONS,
    KEY_RESPONSES,
    KEY_SLOTS,
    DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
    DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
)


class SessionConfig(NamedTuple):
    """The Session Configuration."""

    session_expiration_time: float  # in minutes
    carry_over_slots: bool

    @staticmethod
    def default() -> "SessionConfig":
        """Returns the SessionConfig with the default values."""
        return SessionConfig(
            DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
            DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION,
        )

    def are_sessions_enabled(self) -> bool:
        """Returns a boolean value depending on the value of session_expiration_time."""
        return self.session_expiration_time > 0

    def as_dict(self) -> Dict:
        """Return serialized `SessionConfig`."""
        return {
            "session_expiration_time": self.session_expiration_time,
            "carry_over_slots_to_new_session": self.carry_over_slots,
        }


def extract_duplicates(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Extracts duplicates from two lists."""
    if list1:
        dict1 = {
            (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in list1
        }
    else:
        dict1 = {}

    if list2:
        dict2 = {
            (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in list2
        }
    else:
        dict2 = {}

    set1 = set(dict1.keys())
    set2 = set(dict2.keys())
    dupes = set1.intersection(set2)
    return sorted(list(dupes))


def clean_duplicates(dupes: Dict[Text, Any]) -> Dict[Text, Any]:
    """Removes keys for empty values."""
    duplicates = dupes.copy()
    for k in dupes:
        if not dupes[k]:
            duplicates.pop(k)

    return duplicates


def merge_dicts(
    tempDict1: Dict[Text, Any],
    tempDict2: Dict[Text, Any],
    override_existing_values: bool = False,
) -> Dict[Text, Any]:
    """Merges two dicts."""
    if override_existing_values:
        merged_dicts, b = tempDict1.copy(), tempDict2.copy()

    else:
        merged_dicts, b = tempDict2.copy(), tempDict1.copy()
    merged_dicts.update(b)
    return merged_dicts


def merge_lists(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Merges two lists."""
    return sorted(list(set(list1 + list2)))


def merge_lists_of_dicts(
    dict_list1: List[Dict],
    dict_list2: List[Dict],
    override_existing_values: bool = False,
) -> List[Dict]:
    """Merges two dict lists."""
    dict1 = {
        (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in dict_list1
    }
    dict2 = {
        (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in dict_list2
    }
    merged_dicts = merge_dicts(dict1, dict2, override_existing_values)
    return list(merged_dicts.values())


def combine_domain_dicts(
    domain_dict: Dict, combined: Dict, override: bool = False, is_dir: bool = False
) -> Dict:
    """Combines two domain dictionaries."""
    if override:
        config = domain_dict["config"]
        for key, val in config.items():
            combined["config"][key] = val

    if (
        override
        or (
            not is_dir
            and combined.get(SESSION_CONFIG_KEY) == SessionConfig.default().as_dict()
        )
        or (is_dir and domain_dict.get(SESSION_CONFIG_KEY))
    ):
        combined[SESSION_CONFIG_KEY] = domain_dict[SESSION_CONFIG_KEY]

    duplicates: Dict[Text, List[Text]] = {}

    for key in [KEY_INTENTS, KEY_ENTITIES]:
        if combined.get(key) or domain_dict.get(key):
            duplicates[key] = extract_duplicates(
                combined.get(key, []), domain_dict.get(key, [])
            )
            combined[key] = combined.get(key, [])
            domain_dict[key] = domain_dict.get(key, [])
            combined[key] = merge_lists_of_dicts(
                combined[key], domain_dict[key], override
            )

    # remove existing forms from new actions
    for form in combined.get(KEY_FORMS, []):
        if form in domain_dict.get(KEY_ACTIONS, []):
            domain_dict[KEY_ACTIONS].remove(form)

    for key in [KEY_ACTIONS, KEY_E2E_ACTIONS]:
        duplicates[key] = extract_duplicates(
            combined.get(key, []), domain_dict.get(key, [])
        )
        combined[key] = merge_lists(combined.get(key, []), domain_dict.get(key, []))

    for key in [KEY_FORMS, KEY_RESPONSES, KEY_SLOTS]:
        duplicates[key] = extract_duplicates(
            combined.get(key, []), domain_dict.get(key, [])
        )
        combined[key] = merge_dicts(
            combined.get(key, {}), domain_dict.get(key, {}), override
        )

    if duplicates:
        duplicates = clean_duplicates(duplicates)
        combined.update({"duplicates": duplicates})

    return combined
