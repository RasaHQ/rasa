import abc
from typing import Optional, List


class Intent:
    def __init__(self, name: str, examples: List[str]):
        self.name = name
        self.examples = examples


class Action(abc.ABC):
    @property
    def name(self) -> str:
        pass


class Utterance(Action):

    # _name: str
    # text: str

    def __init__(self, text: str, name: Optional[str] = None):
        self.text = text
        self._name = name

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
