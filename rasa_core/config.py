import os
import typing
from typing import Optional, Text, List

from rasa_core import utils
from rasa_core.policies import PolicyEnsemble

if typing.TYPE_CHECKING:
    from rasa_core.policies import Policy


def load(config_file: Optional[Text]) -> List['Policy']:
    """Load policy data stored in the specified file."""

    if config_file and os.path.isfile(config_file):
        config_data = utils.read_yaml_file(config_file)
    else:
        raise ValueError("You have to provide a valid path to a config file. "
                         "The file '{}' could not be found."
                         "".format(os.path.abspath(config_file)))

    return PolicyEnsemble.from_dict(config_data)
