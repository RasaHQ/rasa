import logging
import os
from pathlib import Path
from typing import Text, Optional, Dict, List, Union

from rasa import data
import rasa.utils.io as io_utils
from rasa.core.domain import Domain
from rasa.core.training.story_reader.markdown_story_reader import MarkdownStoryReader
from rasa.core.training.story_reader.story_reader import StoryReader
from rasa.core.training.story_reader.yaml_story_reader import YAMLStoryReader
from rasa.core.training.structures import StoryStep
from rasa.data import YAML_FILE_EXTENSIONS, MARKDOWN_FILE_EXTENSIONS

logger = logging.getLogger(__name__)


def _get_reader(
    filename: Text,
    domain: Domain,
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
) -> StoryReader:

    if data.is_likely_markdown_file(filename):
        return MarkdownStoryReader(domain, template_variables, use_e2e, filename)
    elif data.is_likely_yaml_file(filename):
        return YAMLStoryReader(domain, template_variables, use_e2e, filename)
    else:
        # This is a use case for uploading the story over REST API.
        # The source file has a random name.
        return _guess_reader(filename, domain, template_variables, use_e2e)


def _guess_reader(
    filename: Text,
    domain: Domain,
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
) -> StoryReader:
    if YAMLStoryReader.is_yaml_story_file(filename):
        return YAMLStoryReader(domain, template_variables, use_e2e, filename)
    elif MarkdownStoryReader.is_markdown_story_file(filename):
        return MarkdownStoryReader(domain, template_variables, use_e2e, filename)
    raise ValueError(
        f"Failed to find a reader class for the story file `{filename}`. "
        f"Supported formats are "
        f"{', '.join(MARKDOWN_FILE_EXTENSIONS.union(YAML_FILE_EXTENSIONS))}."
    )


async def load_data_from_resource(
    resource: Union[Text, Path],
    domain: Domain,
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> List["StoryStep"]:
    """Loads core training data from the specified folder.

    Args:
        resource: Folder/File with core training data files.
        domain: Domain object.
        template_variables: Variables that have to be replaced in the training data.
        use_e2e: Identifies if the e2e reader should be used.
        exclusion_percentage: Identifies the percentage of training data that
                              should be excluded from the training.

    Returns:
        Story steps from the training data.
    """
    if not os.path.exists(resource):
        raise ValueError(f"Resource '{resource}' does not exist.")

    return await load_data_from_files(
        io_utils.list_files(resource),
        domain,
        template_variables,
        use_e2e,
        exclusion_percentage,
    )


async def load_data_from_files(
    story_files: List[Text],
    domain: Domain,
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> List["StoryStep"]:
    """Loads core training data from the specified files.

    Args:
        story_files: List of files with training data in it.
        domain: Domain object.
        template_variables: Variables that have to be replaced in the training data.
        use_e2e: Identifies whether the e2e reader should be used.
        exclusion_percentage: Identifies the percentage of training data that
                              should be excluded from the training.

    Returns:
        Story steps from the training data.
    """
    story_steps = []

    for story_file in story_files:

        reader = _get_reader(story_file, domain, template_variables, use_e2e)

        steps = await reader.read_from_file(story_file)
        story_steps.extend(steps)

    if exclusion_percentage and exclusion_percentage != 100:
        import random

        idx = int(round(exclusion_percentage / 100.0 * len(story_steps)))
        random.shuffle(story_steps)
        story_steps = story_steps[:-idx]

    return story_steps
