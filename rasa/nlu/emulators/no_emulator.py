from typing import Any, Dict, Text
from rasa.nlu.emulators.emulator import Emulator


class NoEmulator(Emulator):
    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data to target format."""

        raise NotImplementedError
