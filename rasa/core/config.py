import copy
import os
from typing import Optional, Text, List, Dict, Union, Tuple, Any, TYPE_CHECKING

from rasa.shared.exceptions import FileNotFoundException
import rasa.shared.utils.io
import rasa.shared.utils.cli
from rasa.core.constants import (
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_CORE_FALLBACK_THRESHOLD,
    DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
)
from rasa.shared.core.constants import (
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
)
import rasa.utils.io
from rasa.shared.constants import (
    DEFAULT_NLU_FALLBACK_INTENT_NAME,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier

if TYPE_CHECKING:
    from rasa.core.policies.policy import Policy
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.training_data.structures import StoryStep


def load(config_file: Optional[Union[Text, Dict]]) -> List["Policy"]:
    """Load policy data stored in the specified file."""
    from rasa.core.policies.ensemble import PolicyEnsemble

    if not config_file:
        raise FileNotFoundException(
            f"The provided configuration file path does not seem to be valid. "
            f"The file '{os.path.abspath(config_file)}' could not be found."
        )

    config_data = {}
    if isinstance(config_file, str) and os.path.isfile(config_file):
        config_data = rasa.shared.utils.io.read_config_file(config_file)
    elif isinstance(config_file, Dict):
        config_data = config_file

    return PolicyEnsemble.from_dict(config_data)


def migrate_fallback_policies(config: Dict) -> Tuple[Dict, Optional["StoryStep"]]:
    """Migrate the deprecated fallback policies to their `RulePolicy` counterpart.

    Args:
        config: The model configuration containing deprecated policies.

    Returns:
        The updated configuration and the required fallback rules.
    """
    new_config = copy.deepcopy(config)
    policies = new_config.get("policies", [])

    fallback_config = _get_config_for_name(
        FallbackPolicy.__name__, policies
    ) or _get_config_for_name(TwoStageFallbackPolicy.__name__, policies)

    if not fallback_config:
        return config, None

    rasa.shared.utils.cli.print_info(f"Migrating the '{fallback_config.get('name')}'.")

    _update_rule_policy_config_for_fallback(policies, fallback_config)
    _update_fallback_config(new_config, fallback_config)
    new_config["policies"] = _drop_policy(fallback_config.get("name"), policies)

    # The triggered action is hardcoded for the Two-Stage Fallback`
    fallback_action_name = ACTION_TWO_STAGE_FALLBACK_NAME
    if fallback_config.get("name") == FallbackPolicy.__name__:
        fallback_action_name = fallback_config.get(
            "fallback_action_name", ACTION_DEFAULT_FALLBACK_NAME
        )

    fallback_rule = _get_faq_rule(
        f"Rule to handle messages with low NLU confidence "
        f"(automated conversion from '{fallback_config.get('name')}')",
        DEFAULT_NLU_FALLBACK_INTENT_NAME,
        fallback_action_name,
    )

    return new_config, fallback_rule


def _get_config_for_name(component_name: Text, config_part: List[Dict]) -> Dict:
    return next(
        (config for config in config_part if config.get("name") == component_name), {}
    )


def _update_rule_policy_config_for_fallback(
    policies: List[Dict], fallback_config: Dict
) -> None:
    """Update the `RulePolicy` configuration with the parameters for the fallback.

    Args:
        policies: The current list of configured policies.
        fallback_config: The configuration of the deprecated fallback configuration.
    """
    rule_policy_config = _get_config_for_name(RulePolicy.__name__, policies)

    if not rule_policy_config:
        rule_policy_config = {"name": RulePolicy.__name__}
        policies.append(rule_policy_config)

    core_threshold = fallback_config.get(
        "core_threshold", DEFAULT_CORE_FALLBACK_THRESHOLD
    )
    fallback_action_name = fallback_config.get(
        "fallback_core_action_name"
    ) or fallback_config.get("fallback_action_name", ACTION_DEFAULT_FALLBACK_NAME)

    rule_policy_config.setdefault("core_fallback_threshold", core_threshold)
    rule_policy_config.setdefault("core_fallback_action_name", fallback_action_name)


def _update_fallback_config(config: Dict, fallback_config: Dict) -> None:
    fallback_classifier_config = _get_config_for_name(
        FallbackClassifier.__name__, config.get("pipeline", [])
    )

    if not fallback_classifier_config:
        fallback_classifier_config = {"name": FallbackClassifier.__name__}
        config["pipeline"].append(fallback_classifier_config)

    nlu_threshold = fallback_config.get("nlu_threshold", DEFAULT_NLU_FALLBACK_THRESHOLD)
    ambiguity_threshold = fallback_config.get(
        "ambiguity_threshold", DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD
    )

    fallback_classifier_config.setdefault("threshold", nlu_threshold)
    fallback_classifier_config.setdefault("ambiguity_threshold", ambiguity_threshold)


def _get_faq_rule(rule_name: Text, intent: Text, action_name: Text) -> "StoryStep":
    faq_rule = f"""
       version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

       rules:
       - rule: {rule_name}
         steps:
         - intent: {intent}
         - action: {action_name}
    """

    story_reader = YAMLStoryReader()
    return story_reader.read_from_string(faq_rule)[0]


def _drop_policy(policy_to_drop: Text, policies: List[Dict]) -> List[Dict]:
    return [policy for policy in policies if policy.get("name") != policy_to_drop]


def migrate_mapping_policy_to_rules(
    config: Dict[Text, Any], domain: "Domain"
) -> Tuple[Dict[Text, Any], "Domain", List["StoryStep"]]:
    """Migrate `MappingPolicy` to its `RulePolicy` counterparts.

    This migration will update the config, domain and generate the required rules.

    Args:
        config: The model configuration containing deprecated policies.
        domain: The domain which potentially includes intents with the `triggers`
            property.

    Returns:
        The updated model configuration, the domain without trigger intents, and the
        generated rules.
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

    rasa.shared.utils.cli.print_info(f"Migrating the '{MappingPolicy.__name__}'.")
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
            new_rules.append(trigger_rule)

    # finally update the policies
    policies = _drop_policy(MappingPolicy.__name__, policies)

    if new_rules and not has_rule_policy:
        policies.append({"name": RulePolicy.__name__})
    new_config["policies"] = policies

    return new_config, new_domain, new_rules
