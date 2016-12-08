def test_luis_request():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    norm = em.normalise_request_json({"q": ["arb text"]})
    assert norm == {"text": "arb text"}


def test_luis_response():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    data = {"text": "I want italian food", "intent": "inform", "entities": [{"entity": "cuisine", "value": "italian"}]}
    norm = em.normalise_response_json(data)
    assert norm == {
        "query": data["text"],
        "topScoringIntent": {
            "intent": "inform",
            "score": None
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
    assert norm == {"text": "arb text"}


def test_wit_response():
    from rasa_nlu.emulators.wit import WitEmulator
    em = WitEmulator()
    data = {
        "text": "I want italian food",
        "intent": "inform",
        "entities": [{"entity": "cuisine", "value": "italian", "start": 7, "end": 14}]}
    norm = em.normalise_response_json(data)
    assert norm == [
        {'entities': {'cuisine': {'confidence': None, 'type': 'value', 'value': 'italian', 'start': 7, 'end': 14}},
         'confidence': None, 'intent': 'inform', '_text': 'I want italian food'}]


def test_dummy_request():
    from rasa_nlu.emulators import NoEmulator
    em = NoEmulator()
    norm = em.normalise_request_json({"text": ["arb text"]})
    assert norm == {"text": "arb text"}


def test_dummy_response():
    from rasa_nlu.emulators import NoEmulator
    em = NoEmulator()
    data = {"intent": "greet", "text": "hi", "entities": {}}
    assert em.normalise_response_json(data) == data
