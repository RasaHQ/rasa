from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Optional, Text, Dict, Any, List

from rasa_core import utils
from rasa_core.policies import PolicyEnsemble


def load(config_file):
    # type: (Optional[Text], Dict[Text, Any], int) -> List[Policy]
    """Load policy data stored in the specified file. fallback_args and
    max_history are typically command line arguments. They take precedence
    over the arguments specified in the config yaml.
    """

    config_data = utils.read_yaml_file(config_file)

    return PolicyEnsemble.from_dict(config_data)
