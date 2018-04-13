from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Text

from rasa_core.interpreter import RegexInterpreter
from rasa_core.training.data import DialogueTrainingData
from rasa_core.training.dsl import StoryFileReader
from rasa_core.training.generator import TrainingsDataGenerator
from rasa_core.training.structures import StoryGraph, STORY_END, STORY_START

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.interpreter import NaturalLanguageInterpreter


def extract_story_graph(
        resource_name,  # type: Text
        domain,  # type: Domain
        interpreter=RegexInterpreter()  # type: NaturalLanguageInterpreter
):
    # type: (...) -> StoryGraph

    story_steps = StoryFileReader.read_from_folder(resource_name,
                                                   domain, interpreter)
    return StoryGraph(story_steps)
