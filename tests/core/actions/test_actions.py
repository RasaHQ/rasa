from rasa.shared.core.constants import DEFAULT_ACTION_NAMES, RULE_SNIPPET_ACTION_NAME
from rasa.core.actions.action import default_actions


def test_default_actions_and_names_consistency():
    names_of_default_actions = {action.name() for action in default_actions()}
    names_of_executable_actions_in_constants = set(DEFAULT_ACTION_NAMES) - {
        RULE_SNIPPET_ACTION_NAME
    }
    assert names_of_default_actions == names_of_executable_actions_in_constants
