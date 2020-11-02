def test_luis_request():
    from rasa.nlu.emulators.luis import LUISEmulator

    em = LUISEmulator()
    norm = em.normalise_request_json({"text": ["arb text"]})
    assert norm == {"text": "arb text", "time": None}


def test_luis_response():
    from rasa.nlu.emulators.luis import LUISEmulator

    em = LUISEmulator()
    data = {
        "text": "I want italian food",
        "intent": {"name": "restaurant_search", "confidence": 0.737014589341683},
        "intent_ranking": [
            {"confidence": 0.737014589341683, "name": "restaurant_search"},
            {"confidence": 0.11605464483122209, "name": "goodbye"},
            {"confidence": 0.08816417744097163, "name": "greet"},
            {"confidence": 0.058766588386123204, "name": "affirm"},
        ],
        "entities": [{"entity": "cuisine", "value": "italian", "role": "roleCuisine"}],
    }
    norm = em.normalise_response_json(data)
    assert norm == {
        "query": data["text"],
        "prediction": {
            "normalizedQuery": data["text"],
            "topIntent": "restaurant_search",
            "intents": {
                "restaurant_search": {"score": 0.737014589341683},
                "goodbye": {"score": 0.11605464483122209},
                "greet": {"score": 0.08816417744097163},
                "affirm": {"score": 0.058766588386123204},
            },
            "entities": {
                "roleCuisine": ["italian"],
                "$instance": {
                    "role": "roleCuisine",
                    "type": "cuisine",
                    "text": "italian",
                    "startIndex": None,
                    "length": len("italian"),
                    "score": None,
                    "modelTypeId": 1,
                    "modelType": "Entity Extractor",
                },
            },
        },
    }
