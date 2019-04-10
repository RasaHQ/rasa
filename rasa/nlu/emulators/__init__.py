from typing import Any, Dict, Text


class NoEmulator(object):
    def __init__(self) -> None:
        self.name = None

    def normalise_request_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:

        _data = {}
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]

        if not data.get("project"):
            _data["project"] = "default"
        elif type(data["project"]) == list:
            _data["project"] = data["project"][0]
        else:
            _data["project"] = data["project"]

        if data.get("model"):
            if type(data["model"]) == list:
                _data["model"] = data["model"][0]
            else:
                _data["model"] = data["model"]

        _data['time'] = data["time"] if "time" in data else None
        return _data

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data to target format."""

        return data
