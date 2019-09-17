import typing
from typing import Dict, List, Text

if typing.TYPE_CHECKING:
    from rasa.core.events import Event


class Dialogue(object):
    """A dialogue comprises a list of Turn objects"""

    def __init__(self, name: Text, events: List["Event"]) -> None:
        """This function initialises the dialogue with the dialogue name and the event list."""
        self.name = name
        self.events = events

    def __str__(self) -> Text:
        """This function returns the dialogue and turns."""
        return "Dialogue with name '{}' and turns:\n{}".format(
            self.name, "\n\n".join(["\t{}".format(t) for t in self.events])
        )

    def as_dict(self) -> Dict:
        """This function returns the dialogue as a dictionary to assist in serialization"""
        return {"events": [event.as_dict() for event in self.events], "name": self.name}
