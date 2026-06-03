from pathlib import Path
from typing import Text

from rasa.shared.core.domain import Domain, InvalidDomain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.constants import UTTER_PREFIX
from rasa.utils.converter import TrainingDataConverter


OBSOLETE_RESPOND_PREFIX = "respond_"


def normalize_utter_action(action_name: Text) -> Text:
    """Ensure that response names start with `utter_`.

    Args:
        action_name: The name of the response.

    Returns:
        The name of the response, starting with `utter_`.
    """
    return (
        f"{UTTER_PREFIX}{action_name[len(OBSOLETE_RESPOND_PREFIX):]}"
        if action_name.startswith(OBSOLETE_RESPOND_PREFIX)
        else action_name
    )


class StoryResponsePrefixConverter(TrainingDataConverter):
    """Converter responsible for ensuring that retrieval intent actions in stories
    start with `utter_` instead of `respond_`.
    """

    @classmethod
    def filter(cls, source_path: Path) -> bool:
        """Only accept YAML story files.

        Args:
            source_path: Path to a training data file.

        Returns:
            `True` if the given file is a YAML stories file, `False` otherwise.
        """
        return YAMLStoryReader.is_stories_file(source_path)

    @classmethod
    async def convert_and_write(cls, source_path: Path, output_path: Path) -> None:
        """Migrate retrieval intent responses to the new 2.0 format in stories.

        Before 2.0, retrieval intent responses needed to start
        with `respond_`. Now, they need to start with `utter_`.

        Args:
            source_path: the source YAML stories file.
            output_path: Path to the output directory.
        """
        reader = YAMLStoryReader()
        story_steps = reader.read_from_file(source_path)

        for story_step in story_steps:
            for event in story_step.events:
                if isinstance(event, ActionExecuted):
                    event.action_name = normalize_utter_action(event.action_name)

        output_file = cls.generate_path_for_converted_training_data_file(
            source_path, output_path
        )
        YAMLStoryWriter().dump(output_file, story_steps)


class DomainResponsePrefixConverter(TrainingDataConverter):
    """Converter responsible for ensuring that retrieval intent actions in domain
    start with `utter_` instead of `respond_`.
    """

    @classmethod
    def filter(cls, source_path: Path) -> bool:
        """Only accept domain files.

        Args:
            source_path: Path to a domain file.

        Returns:
            `True` if the given file can is a valid domain file, `False` otherwise.
        """
        try:
            Domain.from_path(source_path)
        except InvalidDomain:
            return False
        return True

    @classmethod
    async def convert_and_write(cls, source_path: Path, output_path: Path) -> None:
        """Migrate retrieval intent responses to the new 2.0 format in domains.

        Before 2.0, retrieval intent responses needed to start
        with `respond_`. Now, they need to start with `utter_`.

        Args:
            source_path: The source domain file.
            output_path: Path to the output directory.
        """
        domain = Domain.from_path(source_path)
        domain_dict = domain.as_dict()
        domain_dict["actions"] = [
            normalize_utter_action(action) for action in domain_dict["actions"]
        ]

        new_domain = Domain.from_dict(domain_dict)
        output_file = cls.generate_path_for_converted_training_data_file(
            source_path, output_path
        )
        new_domain.persist(output_file)
