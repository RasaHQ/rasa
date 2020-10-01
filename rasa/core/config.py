import copy
import os
import typing
from typing import Optional, Text, List, Dict, Union, Tuple, Any

import rasa.shared.utils.io
from rasa.shared.core.constants import (
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
)
import rasa.utils.io
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.policies.rule_policy import RulePolicy

if typing.TYPE_CHECKING:
    from rasa.core.policies.policy import Policy
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.training_data.structures import StoryStep


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


def migrate_fallback_policies(config: Dict) -> Tuple[Dict, List["StoryStep"]]:
    from rasa.core.policies.fallback import FallbackPolicy
    from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy

    fallback_config = _get_config_for_name(
        FallbackPolicy.__name__, config.get("policies", [])
    ) or _get_config_for_name(
        TwoStageFallbackPolicy.__name__, config.get("policies", [])
    )

    if not fallback_config:
        return config, []

    _update_rule_policy_config(config, fallback_config)
    _update_fallback_config(config, fallback_config)
    config["policies"] = _drop_policy(
        fallback_config.get("name"), config.get("policies", [])
    )

    # The triggered action is hardcoded for the `TwoStageFallback`
    fallback_action_name = ACTION_TWO_STAGE_FALLBACK_NAME
    if fallback_config.get("name") == FallbackPolicy.__name__:
        fallback_action_name = fallback_config.get(
            "fallback_action_name", ACTION_DEFAULT_FALLBACK_NAME
        )

    fallback_rule = _get_faq_rule(
        f"Rule to handle messages with low NLU confidence "
        f"(automated conversion from '{fallback_config.get('name')}'",
        DEFAULT_NLU_FALLBACK_INTENT_NAME,
        fallback_action_name,
    )

    return config, fallback_rule


def _get_config_for_name(
    component_name: Text, config_part: List[Dict]
) -> Optional[Dict]:
    return next(
        (config for config in config_part if config.get("name") == component_name), {}
    )


def _update_rule_policy_config(config: Dict, fallback_config: Dict) -> None:
    rule_policy_config = _get_config_for_name(
        RulePolicy.__name__, config.get("policies", [])
    )

    if not rule_policy_config:
        rule_policy_config = {"name": RulePolicy.__name__}
        config["policies"].append(rule_policy_config)

    core_threshold = fallback_config.get("core_threshold", 0.3)
    fallback_action_name = fallback_config.get(
        "fallback_core_action_name"
    ) or fallback_config.get("fallback_action_name", ACTION_DEFAULT_FALLBACK_NAME)

    rule_policy_config.setdefault("core_fallback_threshold", core_threshold)
    rule_policy_config.setdefault("core_fallback_action_name", fallback_action_name)


def _update_fallback_config(config: Dict, fallback_config: Dict) -> None:
    from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier

    fallback_classifier_config = _get_config_for_name(
        FallbackClassifier.__name__, config.get("pipeline", [])
    )

    if not fallback_classifier_config:
        fallback_classifier_config = {"name": FallbackClassifier.__name__}
        config["pipeline"].append(fallback_classifier_config)

    nlu_threshold = fallback_config.get("nlu_threshold", 0.3)
    ambiguity_threshold = fallback_config.get("ambiguity_threshold", 0.1)

    fallback_classifier_config.setdefault("threshold", nlu_threshold)
    fallback_classifier_config.setdefault("ambiguity_threshold", ambiguity_threshold)


def _get_faq_rule(
    rule_name: Text, intent: Text, action_name: Text
) -> List["StoryStep"]:
    faq_rule = f"""
       rules:
       - rule: {rule_name}
         steps:
         - intent: {intent}
         - action: {action_name}
       """

    story_reader = YAMLStoryReader()
    return story_reader.read_from_string(faq_rule)


def _drop_policy(policy_to_drop: Text, policies: List[Dict]) -> List[Dict]:
    return [policy for policy in policies if policy.get("name") != policy_to_drop]


def migrate_mapping_policy_to_rules(
    config: Dict[Text, Any], domain: "Domain"
) -> Tuple[Dict[Text, Any], "Domain", List["StoryStep"]]:
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
        return config, domain, []

    new_config = copy.deepcopy(config)
    new_domain = copy.deepcopy(domain)

    new_rules = []
    for intent, properties in new_domain.intent_properties.items():
        # remove triggers from intents, if any
        triggered_action = properties.pop("triggers", None)
        if triggered_action:
            trigger_rule = _get_faq_rule(
                f"Rule to map `{intent}` intent to "
                f"`{triggered_action}` (automatic conversion)",
                intent,
                triggered_action,
            )
            new_rules.append(*trigger_rule)

    # finally update the policies
    policies = _drop_policy(MappingPolicy.__name__, policies)

    if new_rules and not has_rule_policy:
        policies.append({"name": RulePolicy.__name__})
    new_config["policies"] = policies

    return new_config, new_domain, new_rules
