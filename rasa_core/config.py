from rasa_core.policies import PolicyEnsemble
from rasa_core import utils


def load(config_file, fallback_args, max_history):
    # type: (Text, Dict, int) -> List[Policy]
    """Load policy data stored in the specified file."""

    if config_file is None:
        return PolicyEnsemble.default_policies(fallback_args, max_history)

    config_data = utils.read_yaml_file(config_file)

    return PolicyEnsemble.from_dict(config_data)
