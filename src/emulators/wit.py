from rasa_nlu.emulators import NoEmulator


class WitEmulator(NoEmulator):
    def __init__(self):
        super(WitEmulator, self).__init__()
        self.name = "wit"

    def normalise_response_json(self, data):
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
                "confidence": data["confidence"],
                "intent": data["intent"],
                "entities": entities
            }
        ]
