import os
from typing import Text, List, Dict, Union, TYPE_CHECKING

import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.shared.utils.io
import rasa.utils.io
import rasa.utils.io

if TYPE_CHECKING:
    from rasa.core.policies.policy import Policy


def load(config_file: Union[Text, Dict]) -> List["Policy"]:
    """Load policy data stored in the specified file."""
    from rasa.core.policies.ensemble import PolicyEnsemble

    config_data = {}
    if isinstance(config_file, str) and os.path.isfile(config_file):
        config_data = rasa.shared.utils.io.read_model_configuration(config_file)
    elif isinstance(config_file, Dict):
        config_data = config_file

    return PolicyEnsemble.from_dict(config_data)
