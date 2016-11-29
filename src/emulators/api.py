import uuid
from datetime import datetime


class ApiEmulator(object):
    def __init__(self):
        self.name = 'api'

    def normalise_request_json(self, data):
        _data = {}
        _data["text"] = data['q'][0]
        return _data

    def normalise_response_json(self, data):
        return {
            "id": unicode(uuid.uuid1()),
            "timestamp": datetime.now().isoformat("T"),
            "result": {
                "source": "agent",
                "resolvedQuery": data["text"],
                "action": None,
                "actionIncomplete": None,
                "parameters": {key: val for key, val in data["entities"].items()},
                "contexts": [],
                "metadata": {
                    "intentId": unicode(uuid.uuid1()),
                    "webhookUsed": "false",
                    "intentName": data["intent"]
                },
                "fulfillment": {},
                "score": None,
            },
            "status": {
                "code": 200,
                "errorType": "success"
            },
            "sessionId": unicode(uuid.uuid1())
        }
