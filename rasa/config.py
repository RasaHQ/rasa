import copy
from typing import Any, Dict, List, Optional, Text

def override_defaults(defaults: Optional[Dict[Text, Any]],
                      custom: Optional[Dict[Text, Any]]
                      ) -> Dict[Text, Any]:
    """Load the default config of a class and override it with custom config."""

    if defaults:
        config = copy.deepcopy(defaults)
    else:
        config = {}

    if custom:
        config.update(custom)
    return config