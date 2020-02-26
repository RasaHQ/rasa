def test_wit_request():
    from rasa.nlu.emulators.wit import WitEmulator

    em = WitEmulator()
    norm = em.normalise_request_json({"text": ["arb text"]})
    assert norm == {"text": "arb text", "time": None}


def test_wit_response():
    from rasa.nlu.emulators.wit import WitEmulator

    em = WitEmulator()
    data = {
        "text": "I want italian food",
        "intent": {"name": "inform", "confidence": 0.4794813722432127},
        "entities": [{"entity": "cuisine", "value": "italian", "start": 7, "end": 14}],
    }
    norm = em.normalise_response_json(data)
    assert norm == [
        {
            "entities": {
                "cuisine": {
                    "confidence": None,
                    "type": "value",
                    "value": "italian",
                    "start": 7,
                    "end": 14,
                }
            },
            "intent": "inform",
            "_text": "I want italian food",
            "confidence": 0.4794813722432127,
        }
    ]
