from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Text, Union, Optional

from ruamel import yaml
import json
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, LiteralScalarString

import rasa.shared.utils.io
import rasa.shared.core.constants
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
import rasa.shared.core.events
from rasa.shared.core.constants import FLOW_HASHES_SLOT
from rasa.shared.core.events import (
    UserUttered,
    ActionExecuted,
    SlotSet,
    ActiveLoop,
    Event,
    DialogueStackUpdated,
)

from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    KEY_STORIES,
    KEY_STORY_NAME,
    KEY_USER_INTENT,
    KEY_ENTITIES,
    KEY_ACTION,
    KEY_STEPS,
    KEY_CHECKPOINT,
    KEY_SLOT_NAME,
    KEY_CHECKPOINT_SLOTS,
    KEY_OR,
    KEY_USER_MESSAGE,
    KEY_ACTIVE_LOOP,
    KEY_BOT_END_TO_END_MESSAGE,
    KEY_RULES,
    KEY_RULE_FOR_CONVERSATION_START,
    KEY_WAIT_FOR_USER_INPUT_AFTER_RULE,
    KEY_RULE_CONDITION,
    KEY_RULE_NAME,
    KEY_STACK_UPDATE,
    KEY_COMMANDS,
)

from rasa.shared.core.training_data.story_writer.story_writer import StoryWriter
from rasa.shared.core.training_data.structures import (
    StoryStep,
    Checkpoint,
    STORY_START,
    RuleStep,
)
from rasa.shared.utils.yaml import write_yaml


class YAMLStoryWriter(StoryWriter):
    """Writes Core training data into a file in a YAML format."""

    def dumps(
        self,
        story_steps: List[StoryStep],
        is_appendable: bool = False,
        is_test_story: bool = False,
    ) -> Text:
        """Turns Story steps into an YAML string.

        Args:
            story_steps: Original story steps to be converted to the YAML.
            is_appendable: Specify if result should not contain
                           high level keys/definitions and can be appended to
                           the existing story file.
            is_test_story: Identifies if the stories should be exported in test stories
                           format.

        Returns:
            String with story steps in the YAML format.
        """
        stream = yaml.StringIO()
        self.dump(stream, story_steps, is_appendable, is_test_story)
        return stream.getvalue()

    def dump(
        self,
        target: Union[Text, Path, yaml.StringIO],
        story_steps: List[StoryStep],
        is_appendable: bool = False,
        is_test_story: bool = False,
    ) -> None:
        """Writes Story steps into a target file/stream.

        Args:
            target: name of the target file/stream to write the YAML to.
            story_steps: Original story steps to be converted to the YAML.
            is_appendable: Specify if result should not contain
                           high level keys/definitions and can be appended to
                           the existing story file.
            is_test_story: Identifies if the stories should be exported in test stories
                           format.
        """
        result = self.stories_to_yaml(story_steps, is_test_story)
        if is_appendable and KEY_STORIES in result:
            result = result[KEY_STORIES]
        write_yaml(result, target, True)

    def stories_to_yaml(
        self, story_steps: List[StoryStep], is_test_story: bool = False
    ) -> Dict[Text, Any]:
        """Converts a sequence of story steps into yaml format.

        Args:
            story_steps: Original story steps to be converted to the YAML.
            is_test_story: `True` if the story is an end-to-end conversation test story.
        """
        from rasa.shared.utils.yaml import KEY_TRAINING_DATA_FORMAT_VERSION

        self._is_test_story = is_test_story

        stories = []
        rules = []
        for story_step in story_steps:
            if isinstance(story_step, RuleStep):
                rules.append(self.process_rule_step(story_step))
            else:
                stories.append(self.process_story_step(story_step))

        result: OrderedDict[Text, Any] = OrderedDict()
        result[KEY_TRAINING_DATA_FORMAT_VERSION] = DoubleQuotedScalarString(
            LATEST_TRAINING_DATA_FORMAT_VERSION
        )

        if stories:
            result[KEY_STORIES] = stories
        if rules:
            result[KEY_RULES] = rules

        return result

    def process_story_step(self, story_step: StoryStep) -> OrderedDict:
        """Converts a single story step into an ordered dict.

        Args:
            story_step: A single story step to be converted to the dict.

        Returns:
            Dict with a story step.
        """
        result: OrderedDict[Text, Any] = OrderedDict()
        result[KEY_STORY_NAME] = story_step.block_name
        steps = self.process_checkpoints(story_step.start_checkpoints)
        for event in story_step.events:
            if not self._filter_event(event):
                continue
            processed = self.process_event(event)
            if processed:
                steps.append(processed)

        steps.extend(self.process_checkpoints(story_step.end_checkpoints))

        result[KEY_STEPS] = steps

        return result

    def process_event(self, event: Union[Event, List[Event]]) -> Optional[OrderedDict]:
        """Process an event or list of events."""
        if isinstance(event, list):
            return self.process_or_utterances(event)
        if isinstance(event, UserUttered):
            return self.process_user_utterance(event, self._is_test_story)
        if isinstance(event, ActionExecuted):
            return self.process_action(event)
        if isinstance(event, SlotSet):
            return self.process_slot(event)
        if isinstance(event, ActiveLoop):
            return self.process_active_loop(event)
        if isinstance(event, DialogueStackUpdated):
            return self.process_stack(event)
        return None

    @staticmethod
    def stories_contain_loops(stories: List[StoryStep]) -> bool:
        """Checks if the stories contain at least one active loop.

        Args:
            stories: Stories steps.

        Returns:
            `True` if the `stories` contain at least one active loop.
            `False` otherwise.
        """
        return any(
            [
                [event for event in story_step.events if isinstance(event, ActiveLoop)]
                for story_step in stories
            ]
        )

    @staticmethod
    def process_user_utterance(
        user_utterance: UserUttered, is_test_story: bool = False
    ) -> OrderedDict:
        """Converts a single user utterance into an ordered dict.

        Args:
            user_utterance: Original user utterance object.
            is_test_story: Identifies if the user utterance should be added
                           to the final YAML or not.

        Returns:
            Dict with a user utterance.
        """
        result = CommentedMap()
        if user_utterance.intent_name and not user_utterance.use_text_for_featurization:
            result[KEY_USER_INTENT] = (
                user_utterance.full_retrieval_intent_name
                if user_utterance.full_retrieval_intent_name
                else user_utterance.intent_name
            )

        entities = []
        if len(user_utterance.entities) and not is_test_story:
            for entity in user_utterance.entities:
                if "value" in entity:
                    if hasattr(user_utterance, "inline_comment_for_entity"):
                        # FIXME: to fix this type issue, WronglyClassifiedUserUtterance
                        # needs to be imported but it's currently outside
                        # of `rasa.shared`
                        for predicted in user_utterance.predicted_entities:  # type: ignore[attr-defined]
                            if predicted["start"] == entity["start"]:
                                commented_entity = (
                                    user_utterance.inline_comment_for_entity(
                                        predicted, entity
                                    )
                                )
                                if commented_entity:
                                    entity_map = CommentedMap(
                                        [(entity["entity"], entity["value"])]
                                    )
                                    entity_map.yaml_add_eol_comment(
                                        commented_entity, entity["entity"]
                                    )
                                    entities.append(entity_map)
                                else:
                                    entities.append(
                                        OrderedDict(
                                            [(entity["entity"], entity["value"])]
                                        )
                                    )
                    else:
                        entities.append(
                            OrderedDict([(entity["entity"], entity["value"])])
                        )
                else:
                    entities.append(entity["entity"])
            result[KEY_ENTITIES] = entities

        if hasattr(user_utterance, "inline_comment"):
            # FIXME: to fix this type issue, WronglyClassifiedUserUtterance needs to
            # be imported but it's currently outside of `rasa.shared`
            comment = user_utterance.inline_comment(
                force_comment_generation=not entities
            )
            if comment:
                result.yaml_add_eol_comment(comment, KEY_USER_INTENT)

        if user_utterance.text and (
            # We only print the utterance text if it was an end-to-end prediction
            user_utterance.use_text_for_featurization
            # or if we want to print a conversation test story.
            or is_test_story
        ):
            result[KEY_USER_MESSAGE] = LiteralScalarString(
                rasa.shared.core.events.format_message(
                    user_utterance.text,
                    user_utterance.intent_name,
                    user_utterance.entities,
                )
            )

        if user_utterance.commands:
            result[KEY_COMMANDS] = user_utterance.commands
        return result

    @staticmethod
    def process_action(action: ActionExecuted) -> Optional[OrderedDict]:
        """Converts a single action into an ordered dict.

        Args:
            action: Original action object.

        Returns:
            Dict with an action.
        """
        if action.action_name == rasa.shared.core.constants.RULE_SNIPPET_ACTION_NAME:
            return None

        result = CommentedMap()
        if action.action_name:
            result[KEY_ACTION] = action.action_name
        elif action.action_text:
            result[KEY_BOT_END_TO_END_MESSAGE] = action.action_text

        if hasattr(action, "inline_comment"):
            # FIXME: to fix this type issue, WarningPredictedAction needs to
            # be imported but it's currently outside of `rasa.shared`
            comment = action.inline_comment()
            if KEY_ACTION in result and comment:
                result.yaml_add_eol_comment(comment, KEY_ACTION)
            elif KEY_BOT_END_TO_END_MESSAGE in result and comment:
                result.yaml_add_eol_comment(comment, KEY_BOT_END_TO_END_MESSAGE)

        return result

    @staticmethod
    def process_slot(event: SlotSet) -> Optional[OrderedDict]:
        """Converts a single `SlotSet` event into an ordered dict.

        Args:
            event: Original `SlotSet` event.

        Returns:
            OrderedDict with an `SlotSet` event.
        """
        if event.key == FLOW_HASHES_SLOT:
            # this is a build in slot which should not be persisted
            return None
        return OrderedDict([(KEY_SLOT_NAME, [{event.key: event.value}])])

    @staticmethod
    def process_stack(event: DialogueStackUpdated) -> OrderedDict:
        """Converts a stack event into an ordered dict.

        Args:
            event: Original stack event.

        Returns:
            OrderedDict with a stack event.
        """
        return OrderedDict([(KEY_STACK_UPDATE, json.loads(event.update))])

    @staticmethod
    def process_checkpoints(checkpoints: List[Checkpoint]) -> List[OrderedDict]:
        """Converts checkpoints event into an ordered dict.

        Args:
            checkpoints: List of original checkpoint.

        Returns:
            List of converted checkpoints.
        """
        result = []
        for checkpoint in checkpoints:
            if checkpoint.name == STORY_START:
                continue
            next_checkpoint: OrderedDict[Text, Any] = OrderedDict(
                [(KEY_CHECKPOINT, checkpoint.name)]
            )
            if checkpoint.conditions:
                next_checkpoint[KEY_CHECKPOINT_SLOTS] = [
                    {key: value} for key, value in checkpoint.conditions.items()
                ]
            result.append(next_checkpoint)
        return result

    def process_or_utterances(self, utterances: List[UserUttered]) -> OrderedDict:
        """Converts user utterance containing the `OR` statement.

        Args:
            utterances: User utterances belonging to the same `OR` statement.

        Returns:
            Dict with converted user utterances.
        """
        return OrderedDict(
            [
                (
                    KEY_OR,
                    [
                        self.process_user_utterance(utterance, self._is_test_story)
                        for utterance in utterances
                    ],
                )
            ]
        )

    @staticmethod
    def process_active_loop(event: ActiveLoop) -> OrderedDict:
        """Converts ActiveLoop event into an ordered dict.

        Args:
            event: ActiveLoop event.

        Returns:
            Converted event.
        """
        return OrderedDict([(KEY_ACTIVE_LOOP, event.name)])

    def process_rule_step(self, rule_step: RuleStep) -> OrderedDict:
        """Converts a RuleStep into an ordered dict.

        Args:
            rule_step: RuleStep object.

        Returns:
            Converted rule step.
        """
        result: OrderedDict[Text, Any] = OrderedDict()
        result[KEY_RULE_NAME] = rule_step.block_name

        condition_steps = []
        condition_events = rule_step.get_rules_condition()
        for event in condition_events:
            processed = self.process_event(event)
            if processed:
                condition_steps.append(processed)
        if condition_steps:
            result[KEY_RULE_CONDITION] = condition_steps

        normal_events = rule_step.get_rules_events()
        if normal_events and not (
            isinstance(normal_events[0], ActionExecuted)
            and normal_events[0].action_name
            == rasa.shared.core.constants.RULE_SNIPPET_ACTION_NAME
        ):
            result[KEY_RULE_FOR_CONVERSATION_START] = True

        normal_steps = []
        for event in normal_events:
            processed = self.process_event(event)
            if processed:
                normal_steps.append(processed)
        if normal_steps:
            result[KEY_STEPS] = normal_steps

        if len(normal_events) > 1:
            last_event = normal_events[len(normal_events) - 1]
            if (
                isinstance(last_event, ActionExecuted)
                and last_event.action_name
                == rasa.shared.core.constants.RULE_SNIPPET_ACTION_NAME
            ):
                result[KEY_WAIT_FOR_USER_INPUT_AFTER_RULE] = False

        return result
