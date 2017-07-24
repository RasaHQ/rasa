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

        self.name = None    # type: Optional[Text]

    def normalise_request_json(self, data):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]

        _data = {}
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]
        if not data.get("agent"):
            _data["agent"] = "default"
        elif type(data["agent"]) == list:
            _data["agent"] = data["agent"][0]
        else:
            _data["agent"] = data["agent"]
        _data['time'] = data["time"] if "time" in data else None
        return _data

    def normalise_response_json(self, data):
        # type: (Dict[Text, Any]) -> Any
        """Transform data to target format."""

        return data
