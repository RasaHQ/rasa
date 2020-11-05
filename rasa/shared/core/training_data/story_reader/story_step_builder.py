import logging
from typing import Text, Optional, Dict, Any, List

import rasa.shared.core.training_data.structures
import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_STORIES
from rasa.shared.core.events import UserUttered, Event
from rasa.shared.core.training_data.structures import (
    Checkpoint,
    GENERATED_CHECKPOINT_PREFIX,
    GENERATED_HASH_LENGTH,
    STORY_START,
    StoryStep,
    RuleStep,
)

logger = logging.getLogger(__name__)


class StoryStepBuilder:
    def __init__(
        self, name: Text, source_name: Optional[Text], is_rule: bool = False
    ) -> None:
        self.name = name
        self.source_name = source_name
        self.story_steps = []
        self.current_steps = []
        self.start_checkpoints = []
        self.is_rule = is_rule

    def add_checkpoint(self, name: Text, conditions: Optional[Dict[Text, Any]]) -> None:

        # Depending on the state of the story part this
        # is either a start or an end check point
        if not self.current_steps:
            self.start_checkpoints.append(Checkpoint(name, conditions))
        else:
            if conditions:
                rasa.shared.utils.io.raise_warning(
                    f"End or intermediate checkpoints "
                    f"do not support conditions! "
                    f"(checkpoint: {name})",
                    docs=DOCS_URL_STORIES + "#checkpoints",
                )
            additional_steps = []
            for t in self.current_steps:
                if t.end_checkpoints:
                    tcp = t.create_copy(use_new_id=True)
                    tcp.end_checkpoints = [Checkpoint(name)]
                    additional_steps.append(tcp)
                else:
                    t.end_checkpoints = [Checkpoint(name)]
            self.current_steps.extend(additional_steps)

    def _prev_end_checkpoints(self) -> List[Checkpoint]:
        if not self.current_steps:
            return self.start_checkpoints
        else:
            # makes sure we got each end name only once
            end_names = {e.name for s in self.current_steps for e in s.end_checkpoints}
            return [Checkpoint(name) for name in end_names]

    def add_user_messages(
        self, messages: List[UserUttered], is_used_for_training: bool = True
    ) -> None:
        """Adds next story steps with the user's utterances.

        Args:
            messages: User utterances.
            is_used_for_training: Identifies if the user utterance is a part of
              OR statement. This parameter is used only to simplify the conversation
              from MD story files. Don't use it other ways, because it ends up
              in a invalid story that cannot be user for real training.
              Default value is `False`, which preserves the expected behavior
              of the reader.
        """
        self.ensure_current_steps()

        if len(messages) == 1:
            # If there is only one possible intent, we'll keep things simple
            for t in self.current_steps:
                t.add_user_message(messages[0])
        else:
            # this simplifies conversion between formats, but breaks the logic
            if not is_used_for_training:
                for t in self.current_steps:
                    t.add_events(messages)
                return

            # If there are multiple different intents the
            # user can use the express the same thing
            # we need to copy the blocks and create one
            # copy for each possible message
            prefix = GENERATED_CHECKPOINT_PREFIX + "OR_"
            generated_checkpoint = rasa.shared.core.training_data.structures.generate_id(
                prefix, GENERATED_HASH_LENGTH
            )
            updated_steps = []
            for t in self.current_steps:
                for m in messages:
                    copied = t.create_copy(use_new_id=True)
                    copied.add_user_message(m)
                    copied.end_checkpoints = [Checkpoint(generated_checkpoint)]
                    updated_steps.append(copied)
            self.current_steps = updated_steps

    def add_event_as_condition(self, event: Event) -> None:
        self.add_event(event, True)

    def add_event(self, event, is_condition: bool = False) -> None:
        self.ensure_current_steps()
        for t in self.current_steps:
            # conditions are supported only for the RuleSteps
            if isinstance(t, RuleStep) and is_condition:
                t.add_event_as_condition(event)
            else:
                t.add_event(event)

    def ensure_current_steps(self) -> None:
        completed = [step for step in self.current_steps if step.end_checkpoints]
        unfinished = [step for step in self.current_steps if not step.end_checkpoints]
        self.story_steps.extend(completed)
        if unfinished:
            self.current_steps = unfinished
        else:
            self.current_steps = self._next_story_steps()

    def flush(self) -> None:
        if self.current_steps:
            self.story_steps.extend(self.current_steps)
            self.current_steps = []

    def _next_story_steps(self) -> List[StoryStep]:
        start_checkpoints = self._prev_end_checkpoints()
        if not start_checkpoints:
            start_checkpoints = [Checkpoint(STORY_START)]
        step_class = RuleStep if self.is_rule else StoryStep
        current_turns = [
            step_class(
                block_name=self.name,
                start_checkpoints=start_checkpoints,
                source_name=self.source_name,
            )
        ]
        return current_turns
