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
        "responseId": norm["responseId"],
        "queryResult": {
            "queryText": "I want italian food",
            "action": "inform",
            "parameters": {"cuisine": ["italian"]},
            "fulfillmentText": "",
            "fulfillmentMessages": [],
            "outputContexts": [],
            "intent": {"name": "inform", "displayName": "inform"},
            "intentDetectionConfidence": 0.4794813722432127,
        },
    }
