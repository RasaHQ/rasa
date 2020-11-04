from rasa.nlu.emulators.luis import LUISEmulator


def test_luis_request():
    em = LUISEmulator()
    norm = em.normalise_request_json({"text": ["arb text"]})
    assert norm == {"text": "arb text", "time": None}


def test_luis_response():
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
        "entities": [
            {
                "entity": "cuisine",
                "value": "italian",
                "role": "roleCuisine",
                "extractor": "SpacyEntityExtractor",
                "start": 7,
                "end": 14,
            }
        ],
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
                    "roleCuisine": [
                        {
                            "role": "roleCuisine",
                            "type": "cuisine",
                            "text": "italian",
                            "startIndex": 7,
                            "length": len("italian"),
                            "score": None,
                            "modelType": "SpacyEntityExtractor",
                        }
                    ]
                },
            },
        },
    }


def test_luis_response_without_role():
    em = LUISEmulator()
    data = {
        "text": "I want italian food",
        "intent": {"name": "restaurant_search", "confidence": 0.737014589341683},
        "intent_ranking": [
            {"confidence": 0.737014589341683, "name": "restaurant_search"}
        ],
        "entities": [{"entity": "cuisine", "value": "italian"}],
    }
    norm = em.normalise_response_json(data)
    assert norm == {
        "query": data["text"],
        "prediction": {
            "normalizedQuery": data["text"],
            "topIntent": "restaurant_search",
            "intents": {"restaurant_search": {"score": 0.737014589341683}},
            "entities": {
                "cuisine": ["italian"],
                "$instance": {
                    "cuisine": [
                        {
                            "role": None,
                            "type": "cuisine",
                            "text": "italian",
                            "startIndex": None,
                            "length": None,
                            "score": None,
                            "modelType": None,
                        }
                    ]
                },
            },
        },
    }


def test_luis_response_without_intent_ranking():
    em = LUISEmulator()
    data = {
        "text": "I want italian food",
        "intent": {"name": "restaurant_search", "confidence": 0.737014589341683},
        "entities": [{"entity": "cuisine", "value": "italian", "role": "roleCuisine"}],
    }
    norm = em.normalise_response_json(data)
    assert norm == {
        "query": data["text"],
        "prediction": {
            "normalizedQuery": data["text"],
            "topIntent": "restaurant_search",
            "intents": {"restaurant_search": {"score": 0.737014589341683}},
            "entities": {
                "roleCuisine": ["italian"],
                "$instance": {
                    "roleCuisine": [
                        {
                            "role": "roleCuisine",
                            "type": "cuisine",
                            "text": "italian",
                            "startIndex": None,
                            "length": None,
                            "score": None,
                            "modelType": None,
                        }
                    ]
                },
            },
        },
    }
