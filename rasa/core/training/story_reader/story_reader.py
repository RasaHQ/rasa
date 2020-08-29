import logging
from typing import Optional, Dict, Text, List, Any

from rasa.core.domain import Domain
from rasa.core.events import SlotSet, ActionExecuted, Event
from rasa.core.exceptions import StoryParseError
from rasa.core.training.story_reader.story_step_builder import StoryStepBuilder
from rasa.core.training.structures import StoryStep

logger = logging.getLogger(__name__)


class StoryReader:
    """Helper class to read a story file."""

    def __init__(
        self,
        domain: Optional[Domain] = None,
        template_vars: Optional[Dict] = None,
        use_e2e: bool = False,
        source_name: Text = None,
        unfold_or_utterances: bool = True,
    ) -> None:
        """Constructor for the StoryReader.

        Args:
            domain: Domain object.
            template_vars: Template variables to be replaced.
            use_e2e: Specifies whether to use the e2e parser or not.
            source_name: Name of the training data source.
            unfold_or_utterances: Identifies if the user utterance is a part of
              OR statement. This parameter is used only to simplify the conversation
              from MD story files. Don't use it other ways, because it ends up
              in a invalid story that cannot be user for real training.
              Default value is `True`, which preserves the expected behavior
              of the reader.
        """
        self.story_steps = []
        self.current_step_builder: Optional[StoryStepBuilder] = None
        self.domain = domain
        self.template_variables = template_vars if template_vars else {}
        self.use_e2e = use_e2e
        self.source_name = source_name
        self.unfold_or_utterances = unfold_or_utterances

    async def read_from_file(self, filename: Text) -> List[StoryStep]:
        raise NotImplementedError

    def _add_current_stories_to_result(self):
        if self.current_step_builder:
            self.current_step_builder.flush()
            self.story_steps.extend(self.current_step_builder.story_steps)

    def _new_story_part(self, name: Text, source_name: Text):
        self._add_current_stories_to_result()
        self.current_step_builder = StoryStepBuilder(name, source_name)

    def _new_rule_part(self, name: Text, source_name: Text):
        self._add_current_stories_to_result()
        self.current_step_builder = StoryStepBuilder(name, source_name, is_rule=True)

    def _add_event(self, event_name, parameters):

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
