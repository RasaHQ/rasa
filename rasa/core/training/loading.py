import logging
import os
from typing import Text, Optional, Dict, List, Any

import rasa.utils.io as io_utils
from rasa.core.domain import Domain
from rasa.core.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.training.structures import StoryStep

logger = logging.getLogger(__name__)


def _get_reader_class(filename: Text) -> Optional[Any]:

    from rasa.core.training.story_reader import yaml_story_reader, markdown_story_reader

    module = None
    if filename.endswith(".md"):
        module = markdown_story_reader.MarkdownStoryReader
    elif filename.endswith(".yml"):
        module = yaml_story_reader.YAMLStoryReader

    return module


async def load_data_from_folder(
    folder: Text,
    domain: Domain,
    interpreter: NaturalLanguageInterpreter = RegexInterpreter(),
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
):
    if not os.path.exists(folder):
        raise ValueError(f"File '{folder}' does not exist.")

    story_steps = await load_data_from_files(
        io_utils.list_files(folder),
        domain,
        interpreter,
        template_variables,
        use_e2e,
        exclusion_percentage,
    )
    return story_steps


async def load_data_from_files(
    story_files: List[Text],
    domain: Domain,
    interpreter: NaturalLanguageInterpreter = RegexInterpreter(),
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> List[StoryStep]:
    story_steps = []

    for story_file in story_files:

        reader_class = _get_reader_class(story_file)
        if not reader_class:
            logger.warning(
                f"Couldn't find a story reader for {story_file}, it will be skipped."
            )
            continue

        reader = reader_class(
            interpreter, domain, template_variables, use_e2e, story_file
        )

        steps = await reader.read_from_file(story_file)
        story_steps.extend(steps)

        # if exclusion percentage is not 100
    if exclusion_percentage and exclusion_percentage != 100:
        import random

        idx = int(round(exclusion_percentage / 100.0 * len(story_steps)))
        random.shuffle(story_steps)
        story_steps = story_steps[:-idx]

    return story_steps
