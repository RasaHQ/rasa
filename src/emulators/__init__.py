class NoEmulator(object):
    def __init__(self):
        self.service = None

    def normalise_request_json(self, data):
        _data = {}
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]
        if not data.get("model"):
            _data["model"] = "default"
        else:
            _data["model"] = data["model"][0] if type(data["model"]) == list else data["model"]
        return _data

    def normalise_response_json(self, data):
        return data
