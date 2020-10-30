from typing import Any, Dict, Text
from rasa.nlu.emulators.emulator import Emulator


class NoEmulator(Emulator):
    """Default emulator that is used when no emulator is specified."""

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data to target format."""
        return data
