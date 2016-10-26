

class LUISEmulator(object):
    def __init__(self):
        self.name='luis'

    def normalise_request_json(self,data):
        _data = {}
        _data["text"]=data['q'][0]
        return _data

    def normalise_response_json(self,data):
        return {
          "query": data["text"],
            "topScoringIntent": {
              "intent": "inform",
              "score": None
            },
          "entities": [
            {
              "entity": e[0],
              "type": e[1],
              "startIndex": None,
              "endIndex": None,
              "score": None
            } for e in data["entities"]
          ]
         }        


