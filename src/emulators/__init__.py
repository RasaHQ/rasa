

class NoEmulator(object):
    def __init__(self):
        self.service= None

    def normalise_request_json(self,data):
        return data

    def normalise_response_json(self,data):
        return data
