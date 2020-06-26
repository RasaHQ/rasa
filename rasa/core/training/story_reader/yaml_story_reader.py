import logging
from typing import Dict, Text, List, Any

import rasa.utils.io
from rasa.constants import DOCS_URL_STORIES
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.core.events import UserUttered, SlotSet
from rasa.core.training.story_reader.story_reader import StoryReader
from rasa.core.training.structures import StoryStep
from rasa.data import YAML_FILE_EXTENSIONS
from rasa.utils.common import raise_warning

logger = logging.getLogger(__name__)

KEY_STORIES = "stories"
KEY_PLAIN_STORY_NAME = "story"
KEY_PLAIN_STORY_STEPS = "steps"
KEY_PLAIN_STORY_USER_UTTERANCE = "user"
KEY_SLOT_NAME = "slot"
KEY_SLOT_VALUE = "value"
KEY_ACTION = "action"
KEY_PLAIN_STORY_ENTITIES = "entities"
KEY_CHECKPOINT = "checkpoint"
KEY_CHECKPOINT_SLOTS = "slots"
KEY_METADATA = "metadata"
KEY_OR = "or"


class YAMLStoryReader(StoryReader):
    """Class that reads the core training data in a YAML format

    """

    async def read_from_file(self, filename: Text) -> List[StoryStep]:
        try:
            yaml_content = rasa.utils.io.read_yaml_file(filename)

            stories = yaml_content.get(KEY_STORIES)  # pytype: disable=attribute-error
            if not stories:
                return []

            self._parse_stories(stories)
            self._add_current_stories_to_result()
            return self.story_steps

        except ValueError:
            logger.error(f"Failed to read {filename}, it will be skipped")

        return []

    def _parse_stories(self, stories: List[Dict[Text, Any]]) -> None:
        for story_item in stories:
            if not isinstance(story_item, dict):
                raise_warning(
                    f"Unexpected block found in {self.source_name}: \n"
                    f"{story_item}\n"
                    f"Items under the `{KEY_STORIES}` key must be YAML dictionaries."
                    f"It will be skipped.",
                    docs=DOCS_URL_STORIES,
                )
                continue

            if KEY_PLAIN_STORY_NAME in story_item.keys():
                self._parse_plain_story(story_item)

    def _parse_plain_story(self, story_item: Dict[Text, Any]) -> None:
        story_name = story_item.get(KEY_PLAIN_STORY_NAME, "")
        if not story_name:
            raise_warning(
                f"Issue found in {self.source_name}: \n"
                f"{story_item}\n"
                f"The story has an empty name."
                f"Stories should have a name defined under `{KEY_PLAIN_STORY_NAME}` key."
                "It will be skipped.",
                docs=DOCS_URL_STORIES,
            )

        steps = story_item.get(KEY_PLAIN_STORY_STEPS, "")

        self._new_story_part(story_name, self.source_name)

        for step in steps:
            self._parse_step(step)

    def _parse_step(self, step: Dict[Text, Any]) -> None:

        if KEY_PLAIN_STORY_USER_UTTERANCE in step.keys():
            self._parse_plain_story_user_utterance(step)
        elif KEY_OR in step.keys():
            self._parse_or_statement(step)
        elif KEY_SLOT_NAME in step.keys():
            self._parse_slot(step)
        elif KEY_ACTION in step.keys():
            self._parse_action(step)
        elif KEY_CHECKPOINT in step.keys():
            self._parse_checkpoint(step)
        elif KEY_METADATA in step.keys():
            pass
        else:
            raise_warning(
                f"Issue found in {self.source_name}: \n"
                f"Found an unexpected step in the story description:\n"
                f"{step}\n"
                "It will be skipped.",
                docs=DOCS_URL_STORIES,
            )

    def _parse_plain_story_user_utterance(self, step: Dict[Text, Any]) -> None:

        if self.use_e2e:
            # TODO
            pass
        else:
            utterance = self._parse_raw_user_utterance(step)
            self.current_step_builder.add_user_messages([utterance])

    def _parse_or_statement(self, step):
        utterances = []

        for sub_step in step.get(KEY_OR):
            if KEY_PLAIN_STORY_USER_UTTERANCE in sub_step.keys():
                utterance = self._parse_raw_user_utterance(sub_step)
                if utterance:
                    utterances.append(utterance)
            else:
                raise_warning(
                    f"Issue found in {self.source_name}: \n"
                    f"`OR` statement can only have `{KEY_PLAIN_STORY_USER_UTTERANCE}` "
                    f"as a sub-element. This step will be skipped:\n"
                    f"`{sub_step}`\n",
                    docs=DOCS_URL_STORIES,
                )
                return

        self.current_step_builder.add_user_messages(utterances)

    def _parse_raw_user_utterance(self, step: Dict[Text, Any]) -> UserUttered:

        user_utterance = step.get(KEY_PLAIN_STORY_USER_UTTERANCE, "").strip()

        if not user_utterance:
            raise_warning(
                f"Issue found in {self.source_name}: \n"
                f"User utterance cannot be empty."
                "This story step will be skipped:\n"
                f"{step}",
                docs=DOCS_URL_STORIES,
            )

        raw_entities = step.get(KEY_PLAIN_STORY_ENTITIES, [])

        final_entities = []
        for entity in raw_entities:
            if isinstance(entity, dict):
                for key, value in entity.items():
                    final_entities.append({"entity": key, "value": value})
            else:
                final_entities.append({"entity": entity, "value": ""})

        if not user_utterance.startswith(INTENT_MESSAGE_PREFIX):
            raise_warning(
                f"Issue found in {self.source_name}: \n"
                f'User intent "{user_utterance}" should start with '
                f'"{INTENT_MESSAGE_PREFIX}"',
                docs=DOCS_URL_STORIES,
            )
        else:
            user_utterance = user_utterance[1:]
        intent = {"name": user_utterance, "confidence": 1.0}
        return UserUttered(user_utterance, intent, final_entities)

    def _parse_slot(self, step: Dict[Text, Any]) -> None:

        slot_name = step.get(KEY_SLOT_NAME, "")
        slot_value = step.get(KEY_SLOT_VALUE, "")

        if not slot_name or not slot_value:
            raise_warning(
                f"Issue found in {self.source_name}: \n"
                f"Slots should have a name and a value."
                "This story step will be skipped:\n"
                f"{step}",
                docs=DOCS_URL_STORIES,
            )
            return

        self._add_event(SlotSet.type_name, {slot_name: slot_value})

    def _parse_action(self, step: Dict[Text, Any]) -> None:

        action_name = step.get(KEY_ACTION, "")
        if not action_name:
            raise_warning(
                f"Issue found in {self.source_name}: \n"
                f"Action name cannot be empty."
                "This story step will be skipped:\n"
                f"{step}",
                docs=DOCS_URL_STORIES,
            )
            return

        self._add_event(action_name, {})

    def _parse_checkpoint(self, step) -> None:

        checkpoint_name = step.get(KEY_CHECKPOINT, "")
        slots = step.get(KEY_CHECKPOINT_SLOTS, [])

        slots_dict = {}

        for slot in slots:
            if not isinstance(slot, dict):
                raise_warning(
                    f"Issue found in {self.source_name}: \n"
                    f'Checkpoint "{checkpoint_name}" has a invalid slot:'
                    f"{slots}\n"
                    f"Items under the `{KEY_CHECKPOINT_SLOTS}` key must be YAML dictionaries."
                    f"The checkpoint will be skipped.",
                    docs=DOCS_URL_STORIES,
                )
                return

            for key, value in slot.items():
                slots_dict[key] = value

        self._add_checkpoint(checkpoint_name, slots_dict)

    @staticmethod
    def is_yaml_story_file(file_path: Text) -> bool:
        if not file_path.split(".")[-1] in YAML_FILE_EXTENSIONS:
            return False
        try:
            content = rasa.utils.io.read_yaml_file(file_path)
            if KEY_STORIES in content:
                return True
        except Exception as e:
            # Using broad `Exception` because yaml library is not exposing all Errors
            logger.error(
                f"Tried to check if '{file_path}' is a story file, but failed to "
                f"read it. If this file contains story data, you should "
                f"investigate this error, otherwise it is probably best to "
                f"move the file to a different location. "
                f"Error: {e}"
            )
            return False
