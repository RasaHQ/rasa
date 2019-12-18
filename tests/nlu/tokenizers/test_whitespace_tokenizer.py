from rasa.nlu.constants import CLS_TOKEN
from rasa.nlu.training_data import TrainingData, Message
from tests.nlu import utilities
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


def test_whitespace():

    tk = WhitespaceTokenizer()

    assert [t.text for t in tk.tokenize("Forecast for lunch")] == [
        "Forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]
    assert [t.lemma for t in tk.tokenize("Forecast for lunch")] == [
        "Forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]

    assert [t.start for t in tk.tokenize("Forecast for lunch")] == [0, 9, 13, 19]

    # we ignore .,!?
    assert [t.text for t in tk.tokenize("hey ńöñàśçií how're you?")] == [
        "hey",
        "ńöñàśçií",
        "how",
        "re",
        "you",
        CLS_TOKEN,
    ]

    assert [t.start for t in tk.tokenize("hey ńöñàśçií how're you?")] == [
        0,
        4,
        13,
        17,
        20,
        24,
    ]

    assert [t.text for t in tk.tokenize("привет! 10.000, ńöñàśçií. (how're you?)")] == [
        "привет",
        "10.000",
        "ńöñàśçií",
        "how",
        "re",
        "you",
        CLS_TOKEN,
    ]

    assert [
        t.start for t in tk.tokenize("привет! 10.000, ńöñàśçií. (how're you?)")
    ] == [0, 8, 16, 27, 31, 34, 38]

    # urls are single token
    assert [
        t.text
        for t in tk.tokenize(
            "https://www.google.com/search?client="
            "safari&rls=en&q="
            "i+like+rasa&ie=UTF-8&oe=UTF-8 "
            "https://rasa.com/docs/nlu/"
            "components/#tokenizer-whitespace"
        )
    ] == [
        "https://www.google.com/search?"
        "client=safari&rls=en&q=i+like+rasa&ie=UTF-8&oe=UTF-8",
        "https://rasa.com/docs/nlu/components/#tokenizer-whitespace",
        CLS_TOKEN,
    ]

    assert [
        t.start
        for t in tk.tokenize(
            "https://www.google.com/search?client="
            "safari&rls=en&q="
            "i+like+rasa&ie=UTF-8&oe=UTF-8 "
            "https://rasa.com/docs/nlu/"
            "components/#tokenizer-whitespace"
        )
    ] == [0, 83, 142]


def test_whitespace_custom_intent_symbol():
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = WhitespaceTokenizer(component_config)

    assert [t.text for t in tk.tokenize("Forecast_for_LUNCH", attribute="intent")] == [
        "Forecast_for_LUNCH"
    ]

    assert [t.text for t in tk.tokenize("Forecast+for+LUNCH", attribute="intent")] == [
        "Forecast",
        "for",
        "LUNCH",
    ]


def test_whitespace_with_case():

    tk = WhitespaceTokenizer()
    assert [t.text for t in tk.tokenize("Forecast for LUNCH")] == [
        "Forecast",
        "for",
        "LUNCH",
        CLS_TOKEN,
    ]

    component_config = {"case_sensitive": False}
    tk = WhitespaceTokenizer(component_config)
    assert [t.text for t in tk.tokenize("Forecast for LUNCH")] == [
        "forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]

    component_config = {"case_sensitive": True}
    tk = WhitespaceTokenizer(component_config)
    assert [t.text for t in tk.tokenize("Forecast for LUNCH")] == [
        "Forecast",
        "for",
        "LUNCH",
        CLS_TOKEN,
    ]

    _config = utilities.base_test_conf("supervised_embeddings")
    examples = [
        Message(
            "Any Mexican restaurant will do",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 4, "end": 11, "value": "Mexican", "entity": "cuisine"}
                ],
            },
        ),
        Message(
            "I want Tacos!",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 7, "end": 12, "value": "Mexican", "entity": "cuisine"}
                ],
            },
        ),
    ]

    component_config = {"case_sensitive": False}
    tk = WhitespaceTokenizer(component_config)
    tk.train(TrainingData(training_examples=examples), _config)
    assert examples[0].data.get("tokens")[0].text == "any"
    assert examples[0].data.get("tokens")[1].text == "mexican"
    assert examples[0].data.get("tokens")[2].text == "restaurant"
    assert examples[0].data.get("tokens")[3].text == "will"
    assert examples[0].data.get("tokens")[4].text == "do"
    assert examples[1].data.get("tokens")[0].text == "i"
    assert examples[1].data.get("tokens")[1].text == "want"
    assert examples[1].data.get("tokens")[2].text == "tacos"
