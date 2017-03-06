def test_luis_request():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    norm = em.normalise_request_json({"q": ["arb text"]})
    assert norm == {"text": "arb text", "model": "default"}


def test_luis_response():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    data = {
        "text": "I want italian food",
        "intent": "inform",
        "confidence": 0.4794813722432127,
        "entities": [{"entity": "cuisine", "value": "italian"}]
    }
    norm = em.normalise_response_json(data)
    assert norm == {
        "query": data["text"],
        "topScoringIntent": {
            "intent": "inform",
            "score": 0.4794813722432127
        },
        "entities": [
            {
                "entity": e["value"],
                "type": e["entity"],
                "startIndex": None,
                "endIndex": None,
                "score": None
            } for e in data["entities"]
            ]
    }


def test_wit_request():
    from rasa_nlu.emulators.wit import WitEmulator
    em = WitEmulator()
    norm = em.normalise_request_json({"q": ["arb text"]})
    assert norm == {"text": "arb text", "model": "default"}


def test_wit_response():
    from rasa_nlu.emulators.wit import WitEmulator
    em = WitEmulator()
    data = {
        "text": "I want italian food",
        "intent": "inform",
        "confidence": 0.4794813722432127,
        "entities": [{"entity": "cuisine", "value": "italian", "start": 7, "end": 14}]}
    norm = em.normalise_response_json(data)
    assert norm == [{
        'entities': {
            'cuisine': {
                'confidence': None,
                'type': 'value',
                'value': 'italian',
                'start': 7,
                'end': 14
            }
        },
        'intent': 'inform',
        '_text': 'I want italian food',
        'confidence': 0.4794813722432127,
    }]


def test_dummy_request():
    from rasa_nlu.emulators import NoEmulator
    em = NoEmulator()
    norm = em.normalise_request_json({"q": ["arb text"]})
    assert norm == {"text": "arb text", "model": "default"}

    norm = em.normalise_request_json({"q": ["arb text"], "model": "specific"})
    assert norm == {"text": "arb text", "model": "specific"}


def test_dummy_response():
    from rasa_nlu.emulators import NoEmulator
    em = NoEmulator()
    data = {"intent": "greet", "text": "hi", "entities": {}, "confidence": 1.0}
    assert em.normalise_response_json(data) == data
