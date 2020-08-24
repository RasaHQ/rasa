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
        "entities": [{"entity": "cuisine", "value": "italian"}],
    }
    norm = em.normalise_response_json(data)
    assert norm == {
        "query": data["text"],
        "topScoringIntent": {"intent": "restaurant_search", "score": 0.737014589341683},
        "intents": [
            {"intent": "restaurant_search", "score": 0.737014589341683},
            {"intent": "goodbye", "score": 0.11605464483122209},
            {"intent": "greet", "score": 0.08816417744097163},
            {"intent": "affirm", "score": 0.058766588386123204},
        ],
        "entities": [
            {
                "entity": e["value"],
                "type": e["entity"],
                "startIndex": None,
                "endIndex": None,
                "score": None,
            }
            for e in data["entities"]
        ],
    }
