from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text


class NoEmulator(object):
    def __init__(self):
        # type: () -> None

        self.name = None  # type: Optional[Text]

    def normalise_request_json(self, data):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]

        _data = {}
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]

        if not data.get("project"):
            _data["project"] = "default"
        elif type(data["project"]) == list:
            _data["project"] = data["project"][0]
        else:
            _data["project"] = data["project"]

        if data.get("model"):
            _data["model"] = data["model"][0] if type(data["model"]) == list else data["model"]

        _data['time'] = data["time"] if "time" in data else None
        return _data

    def normalise_response_json(self, data):
        # type: (Dict[Text, Any]) -> Any
        """Transform data to target format."""

        return data
