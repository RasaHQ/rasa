class WitEmulator(object):
    def __init__(self):
        self.name = "wit"

    def normalise_request_json(self, data):
        _data = {}
        _data["text"] = data["q"][0]
        if "model" not in data:
            _data["model"] = "default"
        else:
            _data["model"] = data["model"][0]
        return _data

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
