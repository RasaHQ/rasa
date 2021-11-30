from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from dataclasses import dataclass
from collections import defaultdict


def utter_entities(dispatcher: CollectingDispatcher, message: Dict[Text, Any]) -> None:
    user_text: Text = message.get("text", "")
    entities: List[Dict[Text, Any]] = message.get("entities")
    for entity in entities:
        entity_type = entity.get("entity")
        entity_value = entity.get("value")
        extractor = entity.get("extractor")
        entity_role = entity.get("role")
        entity_group = entity.get("group")
        start = entity.get("start", 0)
        end = entity.get("end", start)
        user_text_annotated = user_text[:start] + "_" * (end - start) + user_text[end:]
        dispatcher.utter_message(text=f"{user_text_annotated}")
        dispatcher.utter_message(text=f"{entity_type} -> {entity_value} ({entity_role}, {entity_group}), {extractor}")


@dataclass
class Pizza:
    topping: Text = "unknown"
    size: Text = "unknown"
    count: int = 1


def utter_pizzas(dispatcher: CollectingDispatcher, message: Dict[Text, Any]) -> None:
    user_text: Text = message.get("text", "")
    entities: List[Dict[Text, Any]] = message.get("entities")
    pizzas: Dict[Text, Pizza] = defaultdict(Pizza)
    dispatcher.utter_message(text="")
    for entity in entities:
        entity_type = entity.get("entity")
        entity_value: Text = str(entity.get("value"))
        extractor = entity.get("extractor")
        entity_role = entity.get("role")
        entity_group = str(entity.get("group"))

        if entity_type == "pizza":
            pizzas[entity_group].topping = entity_value
        elif entity_type == "number" and entity_role == "count":
            if not entity_value.isnumeric() and entity_value.isdigit():
                dispatcher.utter_message(text=f"{entity_value} is not an integer")
            pizzas[entity_group].count = int(entity_value)
        elif entity_type == "size":
            pizzas[entity_group].size = entity_value
        else:
            dispatcher.utter_message(text=f"Not sure what to make of '{entity_value}'")

    for group, pizza in pizzas.items():
        dispatcher.utter_message(text=f"Pizza {group}: {pizza.count}x {pizza.size} {pizza.topping}")


class TellEntitiesAction(Action):

    def name(self) -> Text:
        return "tell_entities"

    def run(self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        utter_entities(dispatcher, tracker.latest_message)
        intent = tracker.get_intent_of_latest_message()
        if intent == "order_pizza":
            utter_pizzas(dispatcher, tracker.latest_message)
        return []
