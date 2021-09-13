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
        "entities": [
            {
                "entity": "cuisine",
                "value": "italian",
                "start": 7,
                "end": 14,
                "confidence_entity": 0.1234,
            },
            {
                "entity": "cuisine",
                "value": "italian",
                "role": "desert",
                "start": 7,
                "end": 14,
                "confidence_entity": 0.1234,
            },
        ],
    }
    norm = em.normalise_response_json(data)

    expected = {
        "text": "I want italian food",
        "intents": [{"name": "inform", "confidence": 0.4794813722432127}],
        "entities": {
            "cuisine:cuisine": [
                {
                    "name": "cuisine",
                    "role": "cuisine",
                    "start": 7,
                    "end": 14,
                    "body": "italian",
                    "value": "italian",
                    "confidence": 0.1234,
                    "entities": [],
                }
            ],
            "cuisine:desert": [
                {
                    "name": "cuisine",
                    "role": "desert",
                    "start": 7,
                    "end": 14,
                    "body": "italian",
                    "value": "italian",
                    "confidence": 0.1234,
                    "entities": [],
                }
            ],
        },
    }

    assert norm == expected
