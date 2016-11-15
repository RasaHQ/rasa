

def test_luis_request():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    norm = em.normalise_request_json({"q":["arb text"]})
    assert norm == {"text":"arb text"}


def test_luis_response():
    from rasa_nlu.emulators.luis import LUISEmulator
    em = LUISEmulator()
    data = {"text":"I want italian food","intent":"inform","entities":{"cuisine":"italian"}}
    norm = em.normalise_response_json(data)
    assert norm == {
          "query": data["text"],
            "topScoringIntent": {
              "intent": "inform",
              "score": None
            },
          "entities": [
            {
              "entity": e[0],
              "type": e[1],
              "startIndex": None,
              "endIndex": None,
              "score": None
            } for e in data["entities"]
          ]
         }     

def test_wit_request():
    from rasa_nlu.emulators.wit import WitEmulator
    em = WitEmulator()
    norm = em.normalise_request_json({"q":["arb text"]})
    assert norm == {"text":"arb text"}

def test_wit_response():
    assert True    

