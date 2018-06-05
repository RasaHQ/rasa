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
        # type: None -> Text

        # This function returns the dialogue and turns.
        return "Dialogue with name '{}' and turns:\n{}".format(
                self.name, "\n\n".join(["\t{}".format(t) for t in self.events]))

    
class Topic(object):
    """topic of conversation"""

    def __init__(self, name):
        # type: Text -> None

        # The parameter name sets the topic of conversation
        # Passing None leads to 'DefaultTopic'.
        # Similarly, passing 'question' makes a 'QuestionTopic'
        self.name = name


# The default topic will not carry a name nor will it overwrite the topic of
# a dialog e.g. if an action of this default topic is executed, the previous
# topic is kept active
DefaultTopic = Topic(None)

QuestionTopic = Topic("question")
