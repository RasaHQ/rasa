

class LUISEmulator(object):
    def __init__(self):
        self.name='luis'

    def normalise_request_json(self,data):
        pass
    def normalise_response_json(data):
        intent = data['intents'][0]['intent']
        entities = {}
        slots = [e for e in data.get("entities") or []]
        for ent in slots:
            entities[ent["type"]]=ent["entity"]


