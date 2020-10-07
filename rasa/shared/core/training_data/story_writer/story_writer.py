from pathlib import Path
from typing import List, Text, Union, Any, TYPE_CHECKING

from ruamel import yaml

if TYPE_CHECKING:
    from rasa.shared.core.training_data.structures import StoryStep


class StoryWriter:
    @staticmethod
    def dumps(
        story_steps: List["StoryStep"], is_appendable: bool = False, **kwargs: Any
    ) -> Text:
        """Turns Story steps into an string.

        Args:
            story_steps: Original story steps to be converted to the YAML.
            is_appendable: Specify if result should not contain
                           high level keys/definitions and can be appended to
                           the existing story file.
        Returns:
            String with story steps in the desired format.
        """
        raise NotImplementedError

    @staticmethod
    def dump(
        target: Union[Text, Path, yaml.StringIO],
        story_steps: List["StoryStep"],
        is_appendable: bool = False,
    ) -> None:
        """Writes Story steps into a target file/stream.

        Args:
            target: name of the target file/stream to write the string to.
            story_steps: Original story steps to be converted to the string.
            is_appendable: Specify if result should not contain
                           high level keys/definitions and can be appended to
                           the existing story file.
        """
        raise NotImplementedError
