from pathlib import Path
from typing import List, Text, Union

from rasa.shared.constants import DOCS_URL_MIGRATION_GUIDE_MD_DEPRECATION
from ruamel import yaml

from rasa.shared.core.training_data.story_writer.story_writer import StoryWriter
from rasa.shared.core.training_data.structures import StoryStep
import rasa.shared.utils.io


class MarkdownStoryWriter(StoryWriter):
    """Writes Core training data into a file in a markdown format."""

    @staticmethod
    def dump(
        target: Union[Text, Path, yaml.StringIO],
        story_steps: List[StoryStep],
        is_appendable: bool = False,
        is_test_story: bool = False,
    ) -> None:
        """Writes Story steps into a target file/stream.

        Args:
            target: name of the target file/stream to write the string to.
            story_steps: Original story steps to be converted to the string.
            is_appendable: Specify if result should not contain
                           high level keys/definitions and can be appended to
                           the existing story file.
            is_test_story: Identifies if the stories should be exported in test stories
                           format.
        """
        pass

    @staticmethod
    def dumps(
        story_steps: List[StoryStep],
        is_appendable: bool = False,
        is_test_story: bool = False,
        ignore_deprecation_warning: bool = False,
    ) -> Text:
        """Turns Story steps into a markdown string.

        Args:
            story_steps: Original story steps to be converted to the markdown.
            is_appendable: Specify if result should not contain
                           high level keys/definitions and can be appended to
                           the existing story file.
            is_test_story: Identifies if the stories should be exported in test stories
                           format.
            ignore_deprecation_warning: `True` if printing the deprecation warning
                should be suppressed.

        Returns:
            Story steps in the markdown format.
        """
        if not ignore_deprecation_warning:
            rasa.shared.utils.io.raise_deprecation_warning(
                "Stories in Markdown format are deprecated and will be removed in Rasa "
                "Open Source 3.0.0. Please convert your Markdown stories to the "
                "new YAML format.",
                docs=DOCS_URL_MIGRATION_GUIDE_MD_DEPRECATION,
            )
        return MarkdownStoryWriter._stories_to_md(
            story_steps, is_appendable, is_test_story
        )

    @staticmethod
    def _stories_to_md(
        story_steps: List[StoryStep],
        is_appendable: bool = False,
        is_test_story: bool = False,
    ) -> Text:
        story_content = ""
        for step in story_steps:
            story_content += step.as_story_string(is_appendable, is_test_story)
        return story_content.lstrip()
