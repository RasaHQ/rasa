from rasa_nlu.interpreters.simple_interpreter import HelloGoodbyeInterpreter

interpreter = HelloGoodbyeInterpreter()


def test_samples():
    samples = [
        ("Hey there", 
            {'text': "Hey there", 
             'intent': 'greet',
             'entities': [],
             'confidence': 1.0
            }
        ),
        ("good bye for now",
            {'text': "good bye for now",
             'intent': 'goodbye',
             'entities': [],
             'confidence': 1.0
            }
        )
    ]

    for text, result in samples:
        assert interpreter.parse(text) == result, "text : {0} \nresult : {1}, expected {2}".format(text,
                                                                                                   interpreter.parse(
                                                                                                       text), result)
