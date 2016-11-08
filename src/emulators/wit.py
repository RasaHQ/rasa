# data = {'intent':intent,'entities':[slot,val]}

class WitEmulator(object):
    def __init__(self):
        self.name='wit'

    def normalise_request_json(self,data):
        _data = {}
        _data["text"]=data['q'][0]
        return _data

    def normalise_response_json(self,data):
        print('plain response {0}'.format(data))
        return [
          {
            "_text": data["text"],
            "confidence": None,
            "intent": data["intent"],
            "entities" : {key:{"confidence":None,"type":"value","value":val} for key,val in data["entities"].items()}
          }
        ]

