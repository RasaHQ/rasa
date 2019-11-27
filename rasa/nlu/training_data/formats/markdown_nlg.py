import logging
import re
import typing
from typing import Optional, Text, Any, List, Dict

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import TrainingData

from rasa.nlu.training_data.formats.readerwriter import (
    TrainingDataReader,
    TrainingDataWriter,
)


logger = logging.getLogger(__name__)


class NLGMarkdownReader(TrainingDataReader):
    """Reads markdown training data containing NLG stories and creates a TrainingData object."""

    def __init__(self) -> None:
        self.stories = {}

    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Read markdown string and create TrainingData object"""
        from rasa.nlu.training_data import TrainingData

        self.__init__()
        lines = s.splitlines()
        self.stories = self.process_lines(lines)
        return TrainingData(nlg_stories=self.stories)

    @staticmethod
    def process_lines(lines: List[Text]) -> Dict[Text, List[Text]]:

        stories = {}
        story_intent = None
        story_bot_utterances = []  # Keeping it a list for future additions

        for idx, line in enumerate(lines):

            line_num = idx + 1
            try:
                line = line.strip()
                if line == "":
                    continue
                elif line.startswith("#"):
                    # reached a new story block
                    if story_intent:
                        stories[story_intent] = story_bot_utterances
                        story_bot_utterances = []
                        story_intent = None

                elif line.startswith("-"):
                    # reach a assistant's utterance

                    # utterance might have '-' itself, so joining them back if any
                    utterance = "-".join(line.split("- ")[1:])
                    story_bot_utterances.append(utterance)

                elif line.startswith("*"):
                    # reached a user intent
                    story_intent = "*".join(line.split("* ")[1:])

                else:
                    # reached an unknown type of line
                    logger.warning(
                        f"Skipping line {line_num}. "
                        "No valid command found. "
                        f"Line Content: '{line}'"
                    )
            except Exception as e:
                msg = f"Error in line {line_num}: {e}"
                logger.error(msg, exc_info=1)  # pytype: disable=wrong-arg-types
                raise ValueError(msg)

        # add last story
        if story_intent:
            stories[story_intent] = story_bot_utterances

        return stories


class NLGMarkdownWriter(TrainingDataWriter):
    def dumps(self, training_data):
        """Transforms the NlG part of TrainingData object into a markdown string."""
        md = ""
        md += self._generate_nlg_stories(training_data)

        return md

    @staticmethod
    def _generate_nlg_stories(training_data: "TrainingData"):

        md = ""
        for intent, utterances in training_data.nlg_stories.items():
            md += "## \n"
            md += f"* {intent}\n"
            for utterance in utterances:
                md += f"- {utterance}\n"
            md += "\n"
        return md
