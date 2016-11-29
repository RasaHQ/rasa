class NoEmulator(object):
    def __init__(self):
        self.service = None

    def normalise_request_json(self, data):
        for key, val in data.iteritems():
            if type(val) == list:
                data[key] = val[0]
        return data

    def normalise_response_json(self, data):
        return data
