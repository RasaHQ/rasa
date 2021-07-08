import logging
from pathlib import Path
from typing import Optional, Dict, Text, List, Any, Union

import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS
from rasa.shared.core.constants import (
    LEGACY_ACTION_DEACTIVATE_LOOP_NAME,
    ACTION_DEACTIVATE_LOOP_NAME,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet, ActionExecuted, Event
from rasa.shared.exceptions import RasaCoreException
from rasa.shared.core.training_data.story_reader.story_step_builder import (
    StoryStepBuilder,
)
from rasa.shared.core.training_data.structures import StoryStep

logger = logging.getLogger(__name__)


class StoryReader:
    """Helper class to read a story file."""

    def __init__(
        self,
        domain: Optional[Domain] = None,
        template_vars: Optional[Dict] = None,
        use_e2e: bool = False,
        source_name: Optional[Text] = None,
        is_used_for_training: bool = True,
    ) -> None:
        """Constructor for the StoryReader.

        Args:
            domain: Domain object.
            template_vars: Template variables to be replaced.
            use_e2e: Specifies whether to use the e2e parser or not.
            source_name: Name of the training data source.
            is_used_for_training: Identifies if the user utterances should be parsed
              (entities are extracted and removed from the original text) and
              OR statements should be unfolded. This parameter is used only to
              simplify the conversation from MD story files. Don't use it other ways,
              because it ends up in a invalid story that cannot be user for real
              training. Default value is `False`, which preserves the expected behavior
              of the reader.
        """
        self.story_steps = []
        self.current_step_builder: Optional[StoryStepBuilder] = None
        self.domain = domain
        self.template_variables = template_vars if template_vars else {}
        self.use_e2e = use_e2e
        self.source_name = source_name
        self._is_used_for_training = is_used_for_training
        self._is_parsing_conditions = False

    def read_from_file(
        self, filename: Text, skip_validation: bool = False
    ) -> List[StoryStep]:
        """Reads stories or rules from file.

        Args:
            filename: Path to the story/rule file.
            skip_validation: `True` if file validation should be skipped.

        Returns:
            `StoryStep`s read from `filename`.
        """
        raise NotImplementedError

    @staticmethod
    def is_test_stories_file(filename: Text) -> bool:
        """Checks if the specified file is a test story file.

        Args:
            filename: File to check.

        Returns:
            `True` if specified file is a test story file, `False` otherwise.
        """
        raise NotImplementedError

    @staticmethod
    def is_stories_file(filename: Union[Text, Path]) -> bool:
        """Checks if the specified file is a story file.

        Args:
            filename: File to check.

        Returns:
            `True` if specified file is a story file, `False` otherwise.
        """
        raise NotImplementedError

    def _add_current_stories_to_result(self) -> None:
        if self.current_step_builder:
            self.current_step_builder.flush()
            self.story_steps.extend(self.current_step_builder.story_steps)

    def _new_story_part(self, name: Text, source_name: Optional[Text]) -> None:
        self._add_current_stories_to_result()
        self.current_step_builder = StoryStepBuilder(name, source_name)

    def _new_rule_part(self, name: Text, source_name: Optional[Text]) -> None:
        self._add_current_stories_to_result()
        self.current_step_builder = StoryStepBuilder(name, source_name, is_rule=True)

    def _add_event(self, event_name: Text, parameters: Dict[Text, Any]) -> None:
        # add 'name' only if event is not a SlotSet,
        # because there might be a slot with slot_key='name'
        if "name" not in parameters and event_name != SlotSet.type_name:
            parameters["name"] = event_name

        parsed_events = Event.from_story_string(
            event_name, parameters, default=ActionExecuted
        )
        if parsed_events is None:
            raise StoryParseError(
                "Unknown event '{}'. It is Neither an event "
                "nor an action).".format(event_name)
            )
        if self.current_step_builder is None:
            raise StoryParseError(
                "Failed to handle event '{}'. There is no "
                "started story block available. "
                "".format(event_name)
            )

        for p in parsed_events:
            _map_legacy_event_names(p)
            if self._is_parsing_conditions:
                self.current_step_builder.add_event_as_condition(p)
            else:
                self.current_step_builder.add_event(p)

    def _add_checkpoint(
        self, name: Text, conditions: Optional[Dict[Text, Any]]
    ) -> None:

        # Ensure story part already has a name
        if not self.current_step_builder:
            raise StoryParseError(
                "Checkpoint '{}' is at an invalid location. "
                "Expected a story start.".format(name)
            )

        self.current_step_builder.add_checkpoint(name, conditions)


def _map_legacy_event_names(event: Event) -> None:
    if (
        isinstance(event, ActionExecuted)
        and event.action_name == LEGACY_ACTION_DEACTIVATE_LOOP_NAME
    ):
        rasa.shared.utils.io.raise_deprecation_warning(
            f"Using action '{event.action_name}' is deprecated. Please use "
            f"'{ACTION_DEACTIVATE_LOOP_NAME}' instead. Support for "
            f"'{event.action_name}' will be removed in Rasa Open Source version "
            f"{NEXT_MAJOR_VERSION_FOR_DEPRECATIONS}."
        )
        event.action_name = ACTION_DEACTIVATE_LOOP_NAME


class StoryParseError(RasaCoreException, ValueError):
    """Raised if there is an error while parsing a story file."""

    def __init__(self, message: Text) -> None:
        self.message = message
        super(StoryParseError, self).__init__()
