from rasa_core.policies import PolicyEnsemble
from rasa_core import utils


def load(config_file, fallback_args, max_history):
    # type: (Text, Dict, int) -> List[Policy]
    """Load policy data stored in the specified file. fallback_args and
    max_history are typically command line arguments. They take precedence
    over the arguments specified in the config yaml.
    """

    if config_file is None:
        return PolicyEnsemble.default_policies(fallback_args, max_history)

    config_data = utils.read_yaml_file(config_file)
    config_data = handle_precedence(config_data, fallback_args, max_history)

    return PolicyEnsemble.from_dict(config_data)

def handle_precedence(config_data, fallback_args, max_history):
    # type: (Dict, Dict, int) -> Dict

    for policy in config_data:

        if policy.get('name') == 'FallbackPolicy':
            policy.update(fallback_args)

        elif policy.get('name') in {'KerasPolicy', 'MemoizationPolicy'}:
            policy['max_history'] = max_history

    return config_data
