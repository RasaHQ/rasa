import abc
from typing import Optional, List


class Intent:
    def __init__(self, examples: List[str], name: Optional[str] = None):
        if len(examples) == 0:
            raise ValueError("No examples provided.")

        self.examples = examples

        if name:
            self.name = name
        else:
            text_stripped = "".join(
                e.lower() for e in examples[0] if e.isalnum() or e.isspace()
            )
            self.name = "_".join(text_stripped.split(" "))


class Action(abc.ABC):
    @property
    def name(self) -> str:
        pass


class Utterance(Action):

    _name: str
    text: str

    def __init__(self, text: str, name: Optional[str] = None):
        self.text = text

        if name:
            self._name = name
        else:
            text_stripped = "".join(
                e.lower() for e in text if e.isalnum() or e.isspace()
            )
            self._name = "utter_" + "_".join(text_stripped.split(" "))

    @property
    def name(self) -> str:
        return self._name


# class Entity(abc.ABC):
#     @property
#     def name(self) -> str:
#         pass


# enum SpacyEntity: Entity {
#     case GPE
#     case PERSON
#     case LOC

#     var name: str {
#         switch (self) {
#         case .GPE:
#             return "GPE"
#         case .PERSON:
#             return "PERSON"
#         case .LOC:
#             return "LOC"
#         }
#     }
# }


class Slot:
    name: str
    entities: List[str]
    prompt_actions: List[Action]

    def __init__(
        self,
        name: str,
        entities: List[str],
        prompt_actions: List[Action] = [],
    ):
        self.name = name
        self.entities = entities
        self.prompt_actions = prompt_actions
