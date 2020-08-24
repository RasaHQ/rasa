def test_dummy_request():
    from rasa.nlu.emulators.no_emulator import NoEmulator

    em = NoEmulator()
    norm = em.normalise_request_json({"text": ["arb text"]})
    assert norm == {"text": "arb text", "time": None}

    norm = em.normalise_request_json({"text": ["arb text"], "time": "1499279161658"})
    assert norm == {"text": "arb text", "time": "1499279161658"}


def test_dummy_response():
    from rasa.nlu.emulators.no_emulator import NoEmulator

    em = NoEmulator()
    data = {"intent": "greet", "text": "hi", "entities": {}, "confidence": 1.0}
    assert em.normalise_response_json(data) == data


def test_emulators_can_handle_missing_data():
    from rasa.nlu.emulators.luis import LUISEmulator

    em = LUISEmulator()
    norm = em.normalise_response_json(
        {"text": "this data doesn't contain an intent result"}
    )
    assert norm["topScoringIntent"] is None
    assert norm["intents"] == []
