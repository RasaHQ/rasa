from typing import Any, Dict, Text


class NoEmulator:
    def __init__(self) -> None:
        self.name = None

    def normalise_request_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:

        _data = {
            "text": data["text"][0] if type(data["text"]) == list else data["text"]
        }

        if data.get("model"):
            if type(data["model"]) == list:
                _data["model"] = data["model"][0]
            else:
                _data["model"] = data["model"]

        _data["time"] = data["time"] if "time" in data else None
        return _data

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data to target format."""

        return data
