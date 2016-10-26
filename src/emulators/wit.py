# data = {'intent':intent,'entities':[slot,val]}

class WitEmulator(object):
    def __init__(self):
        self.name='wit'

    def normalise_request_json(self,data):
        _data = {}
        data["text"]=_data['q']

    def normalise_response_json(self,data):
        return [
          {
            "_text": data["text"],
            "confidence": null,
            "intent": data["intent"],
            "entities" : {key,{"confidence":null,"type":"value","value":val} for key,val in data["entities"]}
          }
        ]

