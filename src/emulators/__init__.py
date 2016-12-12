class NoEmulator(object):
    def __init__(self):
        self.service = None

    def normalise_request_json(self, data):
        _data = {}
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]
        return _data

    def normalise_response_json(self, data):
        return data
