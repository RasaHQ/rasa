from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.constants import (
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_CORE_FALLBACK_THRESHOLD, DEFAULT_FALLBACK_ACTION)
from rasa_core import utils
from rasa_core.policies import PolicyEnsemble


def load(config_file, fallback_args, max_history):
    # type: Dict[Text, Any] -> List[Policy]
    """Load policy data stored in the specified file. fallback_args and
    max_history are typically command line arguments. They take precedence
    over the arguments specified in the config yaml.
    """

    if config_file is None:
        return PolicyEnsemble.default_policies(fallback_args, max_history)

    config_data = utils.read_yaml_file(config_file)
    config_data = handle_precedence_and_defaults(
                            config_data, fallback_args, max_history)

    return PolicyEnsemble.from_dict(config_data)

def handle_precedence_and_defaults(config_data, fallback_args, max_history):
    # type: Dict[Text, Any] -> Dict[Text, Any]

    for policy in config_data.get('policies'):

        if policy.get('name') == 'FallbackPolicy' and fallback_args is not None:
            set_fallback_args(policy, fallback_args)

        elif policy.get('name') in {'KerasPolicy', 'MemoizationPolicy'}:
            set_arg(policy, "max_history", max_history, 3)

    return config_data

def set_arg(data_dict, argument, value, default):

    if value is not None:
        data_dict[argument] = value
    elif data_dict.get(argument) is None:
        data_dict[argument] = default

    return data_dict

def set_fallback_args(policy, fallback_args):

    set_arg(policy, "nlu_threshold",
            fallback_args.get("nlu_threshold"),
            DEFAULT_NLU_FALLBACK_THRESHOLD)
    set_arg(policy, "core_threshold",
            fallback_args.get("core_threshold"),
            DEFAULT_CORE_FALLBACK_THRESHOLD)
    set_arg(policy, "fallback_action_name",
            fallback_args.get("fallback_action_name"),
            DEFAULT_FALLBACK_ACTION)
