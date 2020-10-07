from pathlib import Path
from typing import List, Text, Union

from ruamel import yaml

from rasa.shared.core.training_data.story_writer.story_writer import StoryWriter
from rasa.shared.core.training_data.structures import StoryStep


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
    ) -> Text:
        """Turns Story steps into a markdown string.

        Args:
            story_steps: Original story steps to be converted to the markdown.
            is_appendable: Specify if result should not contain
                           high level keys/definitions and can be appended to
                           the existing story file.
            is_test_story: Identifies if the stories should be exported in test stories
                           format.

        Returns:
            String with story steps in the markdown format.
        """
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
