from pathlib import Path
from typing import Dict, List, Text, Union

from rasa.shared.utils.io import write_text_file
from rasa.shared.core.training_data.structures import (
    StoryStep,
)

class MarkdownStoryWriter:
    """Writes Core training data into a file in a markdown format."""

    def dumps(self, story_steps: List[StoryStep], flat: bool = False, e2e: bool = False) -> Text:
        """Turns Story steps into a markdown string.

        Args:
            story_steps: Original story steps to be converted to the markdown.
            flat: Specify if result should not contain caption and checkpoints.
        Returns:
            String with story steps in the markdown format.
        """

        return self.stories_to_md(story_steps, flat, e2e)

    def dump(self) -> None:
        raise NotImplementedError

    def stories_to_md(self, story_steps: List[StoryStep], flat, e2e) -> Text:
        story_content = ""
        for step in story_steps:
            story_content += step.as_story_string(flat, e2e)

        return story_content
