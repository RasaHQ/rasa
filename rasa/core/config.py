import os
import typing
from typing import Any, Optional, Text, List, Dict, Union

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.policies.rule_policy import RulePolicy

if typing.TYPE_CHECKING:
    from rasa.core.policies.policy import Policy
    from rasa.shared.core.domain import Domain


def load(config_file: Optional[Union[Text, Dict]]) -> List["Policy"]:
    """Load policy data stored in the specified file."""
    from rasa.core.policies.ensemble import PolicyEnsemble

    if not config_file:
        raise ValueError(
            "You have to provide a valid path to a config file. "
            "The file '{}' could not be found."
            "".format(os.path.abspath(config_file))
        )

    config_data = {}
    if isinstance(config_file, str) and os.path.isfile(config_file):
        config_data = rasa.shared.utils.io.read_config_file(config_file)
    elif isinstance(config_file, Dict):
        config_data = config_file

    return PolicyEnsemble.from_dict(config_data)


def migrate_mapping_policy_to_rules(
    config: Dict[Text, Any], domain: "Domain", rules: List[Dict[Text, Any]]
):
    """
    Migrate MappingPolicy to the new RulePolicy,
    by updating the config, domain and generating rules.

    This function modifies the config, the domain and the rules in place.
    """
    policies = config.get("policies", [])
    has_mapping_policy = False
    has_rule_policy = False

    for policy in policies:
        if policy.get("name") == MappingPolicy.__name__:
            has_mapping_policy = True
        if policy.get("name") == RulePolicy.__name__:
            has_rule_policy = True

    if not has_mapping_policy:
        return

    has_one_triggered_action = False
    for intent, properties in domain.intent_properties.items():
        # remove triggers from intents, if any
        triggered_action = properties.pop("triggers", None)
        if triggered_action:
            has_one_triggered_action = True
            rules.append(
                {
                    "rule": f"Rule to map `{intent}` intent (automatic conversion)",
                    "steps": [
                        {"intent": intent},
                        {"action": triggered_action},
                    ],
                }
            )

    # finally update the policies
    policies = [
        policy for policy in policies if policy.get("name") != MappingPolicy.__name__
    ]
    if has_one_triggered_action and not has_rule_policy:
        policies.append({"name": RulePolicy.__name__})
    config["policies"] = policies
