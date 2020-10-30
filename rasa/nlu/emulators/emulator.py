from typing import Any, Dict, Text


class Emulator:
    """Emulator specifies how requests and responses are getting transformed."""

    @classmethod
    def name(cls) -> Text:
        """Name that identifies the emulator."""
        return cls.__name__

    def normalise_request_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform request JSON to target format.

        Args:
            data: input JSON data as a dictionary.

        Returns:
            The transformed input data.
        """
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
        """Transform response JSON to target format.

        Args:
            data: input JSON data as a dictionary.

        Returns:
            The transformed input data.
        """
        raise NotImplementedError

    def __str__(self) -> Text:
        """Return the string representation of the emulator."""
        return "Emulator('{}')".format(self.name())
