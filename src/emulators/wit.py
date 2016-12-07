class WitEmulator(object):
    def __init__(self):
        self.name = "wit"

    def normalise_request_json(self, data):
        _data = {}
        print data
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]
        return _data

    def normalise_response_json(self, data):
        print('plain response {0}'.format(data))
        entities = {}
        for entity in data["entities"]:
            entities[entity["entity"]] = {"confidence": None, "type": "value", "value": entity["value"]}

        return [
            {
                "_text": data["text"],
                "confidence": None,
                "intent": data["intent"],
                "entities": entities
            }
        ]
