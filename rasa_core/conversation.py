import json
import typing
from typing import List, Text, Optional, Any, Dict

if typing.TYPE_CHECKING:
    from rasa_core.events import Event


class Dialogue(object):
    """A dialogue comprises a list of Turn objects"""

    def __init__(self, name: Text, events: List['Event'],
                 metadata: Optional[Dict[Text, Any]] = None) -> None:
        # This function initialises the dialogue with
        # the dialogue name and the event list.
        self.name = name
        self.events = events
        self.metadata = metadata

    def __str__(self):
        # type: () -> Text

        # This function returns the dialogue and turns.
        return "Dialogue with name '{}', metadata '{}' and turns:\n{}".format(
            self.name, json.dumps(self.metadata),
            "\n\n".join(["\t{}".format(t) for t in self.events]))
