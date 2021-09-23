from pathlib import Path
from rasa.shared.exceptions import FileNotFoundException
from typing import Text, List, Dict, Union, TYPE_CHECKING

import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.utils.io

if TYPE_CHECKING:
    from rasa.core.policies.policy import Policy


def load(config_file: Union[Text, Dict]) -> List["Policy"]:
    """Load policy data stored in the specified file."""
    from rasa.core.policies.ensemble import PolicyEnsemble

    if not config_file:
        raise FileNotFoundException(
            f"The provided configuration file path does not seem to be valid. "
            f"The file '{Path(config_file).resolve()}' could not be found."
        )

    config_data = {}
    if isinstance(config_file, str) and Path(config_file).is_file():
        config_data = rasa.shared.utils.io.read_config_file(config_file)
    elif isinstance(config_file, Dict):
        config_data = config_file

    return PolicyEnsemble.from_dict(config_data)
