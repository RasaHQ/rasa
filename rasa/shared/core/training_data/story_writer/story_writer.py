import typing
from pathlib import Path
from typing import List, Text, Union

from ruamel import yaml

if typing.TYPE_CHECKING:
    from rasa.shared.core.events import Event
    from rasa.shared.core.training_data.structures import StoryStep


class StoryWriter:
    """Writes story training data to file."""

    def dumps(
        self,
        story_steps: List["StoryStep"],
        is_appendable: bool = False,
        is_test_story: bool = False,
    ) -> Text:
        """Turns Story steps into an string.

        Args:
            story_steps: Original story steps to be converted to the YAML.
            is_appendable: Specify if result should not contain
                           high level keys/definitions and can be appended to
                           the existing story file.
            is_test_story: Identifies if the stories should be exported in test stories
                           format.

        Returns:
            String with story steps in the desired format.
        """
        raise NotImplementedError

    def dump(
        self,
        target: Union[Text, Path, yaml.StringIO],
        story_steps: List["StoryStep"],
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
        raise NotImplementedError

    @staticmethod
    def _filter_event(event: Union["Event", List["Event"]]) -> bool:
        """Identifies if the event should be converted/written.

        Args:
            event: target event to check.

        Returns:
            `True` if the event should be converted/written, `False` otherwise.
        """
        from rasa.shared.core.training_data.structures import StoryStep

        # This is an "OR" statement, so we accept it
        if isinstance(event, list):
            return True

        return (
            not StoryStep.is_action_listen(event)
            and not StoryStep.is_action_unlikely_intent(event)
            and not StoryStep.is_action_session_start(event)
        )
