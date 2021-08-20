import logging
import os
from typing import Text, Optional, Dict, List, Union

import rasa.shared.data
import rasa.shared.utils.io
from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.story_reader.story_reader import StoryReader
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.structures import StoryStep
from rasa.shared.data import YAML_FILE_EXTENSIONS

logger = logging.getLogger(__name__)


def _get_reader(
    filename: Text,
    domain: Domain,
    template_variables: Optional[Dict] = None,
) -> StoryReader:
    if rasa.shared.data.is_likely_yaml_file(filename):
        return YAMLStoryReader(domain, template_variables, filename)
    else:
        # This is a use case for uploading the story over REST API.
        # The source file has a random name.
        return _guess_reader(filename, domain, template_variables)


def _guess_reader(
    filename: Text,
    domain: Domain,
    template_variables: Optional[Dict] = None,
) -> StoryReader:
    if YAMLStoryReader.is_stories_file(filename):
        return YAMLStoryReader(domain, template_variables, filename)

    raise ValueError(
        f"Failed to find a reader class for the story file `{filename}`. "
        f"Supported formats are "
        f"{', '.join(YAML_FILE_EXTENSIONS)}."
    )


def load_data_from_resource(
    resource: Union[Text],
    domain: Domain,
    template_variables: Optional[Dict] = None,
    exclusion_percentage: Optional[int] = None,
) -> List["StoryStep"]:
    """Loads core training data from the specified folder.

    Args:
        resource: Folder/File with core training data files.
        domain: Domain object.
        template_variables: Variables that have to be replaced in the training data.
        exclusion_percentage: Identifies the percentage of training data that
                              should be excluded from the training.

    Returns:
        Story steps from the training data.
    """
    if not os.path.exists(resource):
        raise ValueError(f"Resource '{resource}' does not exist.")

    return load_data_from_files(
        rasa.shared.utils.io.list_files(resource),
        domain,
        template_variables,
        exclusion_percentage,
    )


def load_data_from_files(
    story_files: List[Text],
    domain: Domain,
    template_variables: Optional[Dict] = None,
    exclusion_percentage: Optional[int] = None,
) -> List["StoryStep"]:
    """Loads core training data from the specified files.

    Args:
        story_files: List of files with training data in it.
        domain: Domain object.
        template_variables: Variables that have to be replaced in the training data.
        exclusion_percentage: Identifies the percentage of training data that
                              should be excluded from the training.

    Returns:
        Story steps from the training data.
    """
    story_steps = []

    for story_file in story_files:

        reader = _get_reader(story_file, domain, template_variables)

        steps = reader.read_from_file(story_file)
        story_steps.extend(steps)

    if exclusion_percentage and exclusion_percentage != 100:
        import random

        idx = int(round(exclusion_percentage / 100.0 * len(story_steps)))
        random.shuffle(story_steps)
        story_steps = story_steps[:-idx]

    return story_steps
