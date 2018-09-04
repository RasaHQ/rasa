from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import List
from typing import Text

if typing.TYPE_CHECKING:
    from rasa_core.events import Event


class Dialogue(object):
    """A dialogue comprises a list of Turn objects"""

    def __init__(self, name, events):
        # type: (Text, List[Event]) -> None

        # This function initialises the dialogue with
        # the dialogue name and the event list.
        self.name = name
        self.events = events

    def __str__(self):
        # type: () -> Text

        # This function returns the dialogue and turns.
        return "Dialogue with name '{}' and turns:\n{}".format(
                self.name, "\n\n".join(["\t{}".format(t) for t in self.events]))
