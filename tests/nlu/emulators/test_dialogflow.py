def test_dialogflow_request():
    from rasa.nlu.emulators.dialogflow import DialogflowEmulator

    em = DialogflowEmulator()
    norm = em.normalise_request_json({"text": ["arb text"]})
    assert norm == {"text": "arb text", "time": None}


def test_dialogflow_response():
    from rasa.nlu.emulators.dialogflow import DialogflowEmulator

    em = DialogflowEmulator()
    data = {
        "text": "I want italian food",
        "intent": {"name": "inform", "confidence": 0.4794813722432127},
        "entities": [{"entity": "cuisine", "value": "italian", "start": 7, "end": 14}],
    }
    norm = em.normalise_response_json(data)

    assert norm == {
        "id": norm["id"],
        "result": {
            "action": data["intent"]["name"],
            "actionIncomplete": False,
            "contexts": [],
            "fulfillment": {},
            "metadata": {
                "intentId": norm["result"]["metadata"]["intentId"],
                "intentName": data["intent"]["name"],
                "webhookUsed": "false",
            },
            "parameters": {"cuisine": ["italian"]},
            "resolvedQuery": data["text"],
            "score": data["intent"]["confidence"],
            "source": "agent",
        },
        "sessionId": norm["sessionId"],
        "status": {"code": 200, "errorType": "success"},
        "timestamp": norm["timestamp"],
    }
