from rasa_nlu.emulators import NoEmulator


class LUISEmulator(NoEmulator):
    def __init__(self):
        super(LUISEmulator, self).__init__()
        self.name = 'luis'

    def normalise_response_json(self, data):
        return {
            "query": data["text"],
            "topScoringIntent": {
                "intent": data["intent"]["name"],
                "score": data["intent"]["confidence"]
            },
            "entities": [
                {
                    "entity": e["value"],
                    "type": e["entity"],
                    "startIndex": None,
                    "endIndex": None,
                    "score": None
                } for e in data["entities"]
                ]
        }
