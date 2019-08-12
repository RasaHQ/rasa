from rasa_sdk.knowledge_base.storage import InMemoryKnowledgeBase
from rasa_sdk.knowledge_base.actions import ActionQueryKnowledgeBase


SCHEMA = {
    "restaurant": {
        "attributes": ["cuisine", "wifi"],
        "key": "name",
        "representation": lambda e: e["name"],
    },
    "hotel": {
        "attributes": ["breakfast-included", "price-range"],
        "key": "name",
        "representation": lambda e: e["name"] + " (" + e["city"] + ")",
    },
}

DATA = {
    "restaurant": [
        {"name": "PastaBar", "cuisine": "Italian", "wifi": "False"},
        {"name": "Berlin Burrito Company", "cuisine": "Mexican", "wifi": "True"},
        {"name": "I due forni", "cuisine": "Italian", "wifi": "False"},
    ],
    "hotel": [
        {
            "name": "Hilton",
            "price-range": "expensive",
            "breakfast-included": "True",
            "city": "Berlin",
        },
        {
            "name": "Hilton",
            "price-range": "expensive",
            "breakfast-included": "True",
            "city": "Frankfurt am Main",
        },
        {
            "name": "B&B",
            "price-range": "mid-range",
            "breakfast-included": "True",
            "city": "Berlin",
        },
        {
            "name": "Berlin Wall Hostel",
            "price-range": "cheap",
            "breakfast-included": "False",
            "city": "Berlin",
        },
        {
            "name": "City Hotel",
            "price-range": "expensive",
            "breakfast-included": "False",
            "city": "Berlin",
        },
        {
            "name": "Jugendherberge",
            "price-range": "cheap",
            "breakfast-included": "True",
            "city": "Berlin",
        },
        {
            "name": "Berlin Hotel",
            "price-range": "mid-range",
            "breakfast-included": "False",
            "city": "Berlin",
        },
    ],
}


class ActionMyKB(ActionQueryKnowledgeBase):
    def __init__(self):
        knowledge_base = InMemoryKnowledgeBase(SCHEMA, DATA)
        super().__init__(knowledge_base)
