from typing import Any, Dict, List, Text

from rasa_nlu.emulators import NoEmulator


class WitEmulator(NoEmulator):
    def __init__(self) -> None:

        super(WitEmulator, self).__init__()
        self.name = "wit"

    def normalise_response_json(self,
                                data: Dict[Text, Any]
                                ) -> List[Dict[Text, Any]]:
        """Transform data to wit.ai format."""

        entities = {}
        for entity in data["entities"]:
            entities[entity["entity"]] = {
                "confidence": None,
                "type": "value",
                "value": entity["value"],
                "start": entity["start"],
                "end": entity["end"]
            }

        return [
            {
                "_text": data["text"],
                "confidence": data["intent"]['confidence'],
                "intent": data["intent"]['name'],
                "entities": entities
            }
        ]
