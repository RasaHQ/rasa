from typing import Any, Text, Dict, List, Tuple

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
    for entity in entities:
        print(entity)
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


def utter_phone_specs(dispatcher: CollectingDispatcher, message: Dict[Text, Any]) -> None:
    user_text: Text = message.get("text", "")
    entities: List[Dict[Text, Any]] = message.get("entities")
    included_features: List[Text] = []
    excluded_features: List[Text] = []
    for entity in entities:
        entity_type = entity.get("entity")
        entity_value: Text = str(entity.get("value"))
        extractor = entity.get("extractor")
        entity_role = entity.get("role")
        entity_group = str(entity.get("group"))

        if entity_group == "include":
            included_features.append(entity_value)
        elif entity_group == "exclude":
            excluded_features.append(entity_value)
        elif entity_type in ["phone_feature", "color"]:
            included_features.append(entity_value)
        else:
            dispatcher.utter_message(text=f"Not sure what to make of '{entity_value}'")

    dispatcher.utter_message("You want:")
    for feature in included_features:
        dispatcher.utter_message(text=f"- {feature}")
    dispatcher.utter_message("You do not want:")
    for feature in excluded_features:
        dispatcher.utter_message(text=f"- {feature}")


@dataclass
class HomeControlRequirement:
    kind: Text
    which: Text
    value: Text

    def __str__(self) -> str:
        return f"{self.kind}({self.which}) == {self.value}"


@dataclass
class HomeControlAction:
    device: Text
    action: Text

    def __str__(self) -> str:
        return f"{self.device} <- {self.action}"


def utter_home_control_instructions(dispatcher: CollectingDispatcher, message: Dict[Text, Any]) -> None:
    user_text: Text = message.get("text", "")
    entities: List[Dict[Text, Any]] = message.get("entities")
    requirements: List[HomeControlRequirement] = []
    actions: List[HomeControlAction] = []
    parameters: Dict[Text, Any] = dict()
    is_rule: bool = False
    for entity in entities:
        entity_type = entity.get("entity")
        entity_value: Text = str(entity.get("value"))
        extractor = entity.get("extractor")
        entity_role = entity.get("role")
        entity_group = str(entity.get("group"))

        if entity_group == "requirement":
            requirements.append(HomeControlRequirement(entity_type, entity_value, entity_role))
        elif entity_group == "rule_requirement":
            requirements.append(HomeControlRequirement(entity_type, entity_value, entity_role))
            is_rule = True
        elif entity_type == "device" and entity_group == "action":
            actions.append(HomeControlAction(entity_value, entity_role))
        elif entity_type == "number" and entity_role == "celsius" and entity_group == "action":
            parameters[entity_role] = entity_value
        else:
            dispatcher.utter_message(text=f"Not sure what to make of '{entity_value}'")

    conditional = "WHENEVER" if is_rule else "IF"
    req = " and ".join(f"{requirement}" for requirement in requirements)
    act = " and ".join(f"{action}" for action in actions)
    dispatcher.utter_message(text=f"{conditional} {req}")
    dispatcher.utter_message(text=f"THEN {act} ({[f'{val} {role}' for role, val in parameters.items()]})")


class TellEntitiesAction(Action):

    def name(self) -> Text:
        return "tell_entities"

    def run(self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        intent = tracker.get_intent_of_latest_message()
        if intent == "order_pizza":
            utter_pizzas(dispatcher, tracker.latest_message)
        elif intent == "search_phone":
            utter_phone_specs(dispatcher, tracker.latest_message)
        elif intent == "instruct_home_control":
            utter_home_control_instructions(dispatcher, tracker.latest_message)
        else:
            utter_entities(dispatcher, tracker.latest_message)

        return []
