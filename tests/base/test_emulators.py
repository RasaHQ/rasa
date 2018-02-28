from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def test_luis_request():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    norm = em.normalise_request_json({"q": ["arb text"]})
    assert norm == {"text": "arb text", "project": "default", "time": None}


def test_luis_response():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    data = {
        "text": "I want italian food",
        "intent": {"name": "restaurant_search", "confidence": 0.737014589341683},
        "intent_ranking": [
            {
                "confidence": 0.737014589341683,
                "name": "restaurant_search"
            },
            {
                "confidence": 0.11605464483122209,
                "name": "goodbye"
            },
            {
                "confidence": 0.08816417744097163,
                "name": "greet"
            },
            {
                "confidence": 0.058766588386123204,
                "name": "affirm"
            }
        ],
        "entities": [{"entity": "cuisine", "value": "italian"}]
    }
    norm = em.normalise_response_json(data)
    assert norm == {
        "query": data["text"],
        "topScoringIntent": {
            "intent": "restaurant_search",
            "score": 0.737014589341683
        },
        "intents": [
            {
                "intent": "restaurant_search",
                "score": 0.737014589341683
            },
            {
                "intent": "goodbye",
                "score": 0.11605464483122209
            },
            {
                "intent": "greet",
                "score": 0.08816417744097163
            },
            {
                "intent": "affirm",
                "score": 0.058766588386123204
            }
        ],
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
    assert norm == {"text": "arb text", "project": "default", "time": None}


def test_wit_response():
    from rasa_nlu.emulators.wit import WitEmulator
    em = WitEmulator()
    data = {
        "text": "I want italian food",
        "intent": {"name": "inform", "confidence": 0.4794813722432127},
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


def test_dialogflow_request():
    from rasa_nlu.emulators.dialogflow import DialogflowEmulator
    em = DialogflowEmulator()
    norm = em.normalise_request_json({"q": ["arb text"]})
    assert norm == {"text": "arb text", "project": "default", "time": None}


def test_dialogflow_response():
    from rasa_nlu.emulators.dialogflow import DialogflowEmulator
    em = DialogflowEmulator()
    data = {
        "text": "I want italian food",
        "intent": {"name": "inform", "confidence": 0.4794813722432127},
        "entities": [{"entity": "cuisine", "value": "italian", "start": 7, "end": 14}]
    }
    norm = em.normalise_response_json(data)

    assert norm == {
        "id": norm["id"],
        "result": {
            "action": None,
            "actionIncomplete": None,
            "contexts": [],
            "fulfillment": {},
            "metadata": {
                "intentId": norm["result"]["metadata"]["intentId"],
                "intentName": {
                    "confidence": data["intent"]["confidence"],
                    "name": data["intent"]["name"]
                },
                "webhookUsed": "false"
            },
            "parameters": {
                "cuisine": [
                    "italian"
                ]
            },
            "resolvedQuery": data["text"],
            "score": None,
            "source": "agent"
        },
        "sessionId": norm["sessionId"],
        "status": {
            "code": 200,
            "errorType": "success"
        },
        "timestamp": norm["timestamp"]
    }


def test_dummy_request():
    from rasa_nlu.emulators import NoEmulator
    em = NoEmulator()
    norm = em.normalise_request_json({"q": ["arb text"]})
    assert norm == {"text": "arb text", "project": "default", "time": None}

    norm = em.normalise_request_json({"q": ["arb text"], "project": "specific", "time": "1499279161658"})
    assert norm == {"text": "arb text", "project": "specific", "time": "1499279161658"}


def test_dummy_response():
    from rasa_nlu.emulators import NoEmulator
    em = NoEmulator()
    data = {"intent": "greet", "text": "hi", "entities": {}, "confidence": 1.0}
    assert em.normalise_response_json(data) == data


def test_emulators_can_handle_missing_data():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    norm = em.normalise_response_json({"text": "this data doesn't contain an intent result"})
    assert norm["topScoringIntent"] is None
    assert norm["intents"] == []
