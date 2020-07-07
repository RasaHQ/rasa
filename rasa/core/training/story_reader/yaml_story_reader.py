import logging
from pathlib import Path
from typing import Dict, Text, List, Any, Optional, Union

import rasa.utils.common as common_utils
import rasa.utils.io
from rasa.constants import DOCS_URL_STORIES
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.core.events import UserUttered, SlotSet
from rasa.core.training.story_reader.story_reader import StoryReader
from rasa.core.training.structures import StoryStep
from rasa.data import YAML_FILE_EXTENSIONS

logger = logging.getLogger(__name__)

KEY_STORIES = "stories"
KEY_STORY_NAME = "story"
KEY_STORY_STEPS = "steps"
KEY_STORY_USER_INTENT = "intent"
KEY_STORY_USER_END_TO_END_MESSAGE = "user"
KEY_STORY_ENTITIES = "entities"
KEY_SLOT_NAME = "slot"
KEY_SLOT_VALUE = "value"
KEY_ACTION = "action"
KEY_BOT_END_TO_END_MESSAGE = "bot"
KEY_CHECKPOINT = "checkpoint"
KEY_CHECKPOINT_SLOTS = "slots"
KEY_METADATA = "metadata"
KEY_OR = "or"


class YAMLStoryReader(StoryReader):
    """Class that reads the core training data in a YAML format"""

    async def read_from_file(self, filename: Text) -> List[StoryStep]:
        try:
            yaml_content = rasa.utils.io.read_yaml_file(filename)
            if not isinstance(yaml_content, dict):
                common_utils.raise_warning(
                    f"Failed to read '{filename}'. It should be a Yaml dict."
                )
                return []

            return self.read_from_parsed_yaml(yaml_content)

        except ValueError as e:
            common_utils.raise_warning(
                f"Failed to read YAML from '{filename}', it will be skipped. Error: {e}"
            )

        return []

    def read_from_parsed_yaml(
        self, parsed_content: Dict[Text, Union[Dict, List]]
    ) -> List[StoryStep]:
        """Read stories from parsed YAML.

        Args:
            parsed_content: The parsed YAML as Dict.

        Returns:
            The parsed stories.
        """
        stories = parsed_content.get(KEY_STORIES)  # pytype: disable=attribute-error
        if not stories:
            return []

        self._parse_stories(stories)
        self._add_current_stories_to_result()

        return self.story_steps

    def _parse_stories(self, stories: List[Dict[Text, Any]]) -> None:
        for story_item in stories:
            if not isinstance(story_item, dict):
                common_utils.raise_warning(
                    f"Unexpected block found in '{self.source_name}': \n"
                    f"{story_item}\n"
                    f"Items under the '{KEY_STORIES}' key must be YAML dictionaries. "
                    f"It will be skipped.",
                    docs=DOCS_URL_STORIES,
                )
                continue

            if KEY_STORY_NAME in story_item.keys():
                self._parse_plain_story(story_item)

    def _parse_plain_story(self, story_item: Dict[Text, Any]) -> None:
        story_name = story_item.get(KEY_STORY_NAME, "")
        if not story_name:
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"{story_item}\n"
                f"The story has an empty name. "
                f"Stories should have a name defined under '{KEY_STORY_NAME}' key. "
                "It will be skipped.",
                docs=DOCS_URL_STORIES,
            )

        steps = story_item.get(KEY_STORY_STEPS, [])
        if not steps:
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}': "
                f"The story has no steps. "
                "It will be skipped.",
                docs=DOCS_URL_STORIES,
            )
            return

        self._new_story_part(story_name, self.source_name)

        for step in steps:
            self._parse_step(step)

    def _parse_step(self, step: Dict[Text, Any]) -> None:

        if KEY_STORY_USER_INTENT in step.keys():
            self._parse_intent(step)
        elif KEY_STORY_USER_END_TO_END_MESSAGE in step.keys():
            self._parse_user_message(step)
        elif KEY_OR in step.keys():
            self._parse_or_statement(step)
        elif KEY_SLOT_NAME in step.keys():
            self._parse_slot(step)
        elif KEY_ACTION in step.keys():
            self._parse_action(step)
        elif KEY_BOT_END_TO_END_MESSAGE in step.keys():
            self._parse_bot_message(step)
        elif KEY_CHECKPOINT in step.keys():
            self._parse_checkpoint(step)
        elif KEY_METADATA in step.keys():
            pass
        else:
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"Found an unexpected step in the story description:\n"
                f"{step}\n"
                "It will be skipped.",
                docs=DOCS_URL_STORIES,
            )

    def _parse_intent(self, step: Dict[Text, Any]) -> None:
        utterance = self._parse_raw_user_utterance(step)
        if utterance:
            self._validate_that_utterance_is_in_domain(utterance)
            self.current_step_builder.add_user_messages([utterance])

    def _validate_that_utterance_is_in_domain(self, utterance: UserUttered) -> None:
        intent_name = utterance.intent.get("name")

        if not self.domain:
            logger.debug(
                "Skipped validating if intent is in domain as domain " "is `None`."
            )
            return

        if intent_name not in self.domain.intents:
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"Found intent '{intent_name}' in stories which is not part of the "
                f"domain.",
                docs=DOCS_URL_STORIES,
            )

    def _parse_user_message(self, step: Dict[Text, Any]) -> None:
        import rasa.nlu.training_data.entities_parser as entities_parser

        message = step.get(KEY_STORY_USER_END_TO_END_MESSAGE, "")
        entities = entities_parser.find_entities_in_training_example(message)
        plain_text = entities_parser.replace_entities(message)

        self.current_step_builder.add_user_messages(
            [UserUttered(plain_text, {"name": None}, entities=entities)]
        )

    def _parse_or_statement(self, step: Dict[Text, Any]) -> None:
        utterances = []

        for utterance in step.get(KEY_OR):
            if KEY_STORY_USER_INTENT in utterance.keys():
                utterance = self._parse_raw_user_utterance(utterance)
                if utterance:
                    utterances.append(utterance)
            else:
                common_utils.raise_warning(
                    f"Issue found in '{self.source_name}': \n"
                    f"`OR` statement can only have '{KEY_STORY_USER_INTENT}' "
                    f"as a sub-element. This step will be skipped:\n"
                    f"'{utterance}'\n",
                    docs=DOCS_URL_STORIES,
                )
                return

        self.current_step_builder.add_user_messages(utterances)

    def _parse_raw_user_utterance(self, step: Dict[Text, Any]) -> Optional[UserUttered]:
        user_utterance = step.get(KEY_STORY_USER_INTENT, "").strip()

        if not user_utterance:
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"User utterance cannot be empty. "
                f"This story step will be skipped:\n"
                f"{step}",
                docs=DOCS_URL_STORIES,
            )

        raw_entities = step.get(KEY_STORY_ENTITIES, [])
        final_entities = YAMLStoryReader._parse_raw_entities(raw_entities)

        if user_utterance.startswith(INTENT_MESSAGE_PREFIX):
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"User intent '{user_utterance}' starts with "
                f"'{INTENT_MESSAGE_PREFIX}'. This is not required.",
                docs=DOCS_URL_STORIES,
            )
            # Remove leading slash
            user_utterance = user_utterance[1:]

        intent = {"name": user_utterance, "confidence": 1.0}
        return UserUttered(user_utterance, intent, final_entities)

    @staticmethod
    def _parse_raw_entities(
        raw_entities: Union[List[Dict[Text, Text]], List[Text]]
    ) -> List[Dict[Text, Text]]:
        final_entities = []
        for entity in raw_entities:
            if isinstance(entity, dict):
                for key, value in entity.items():
                    final_entities.append({"entity": key, "value": value})
            else:
                final_entities.append({"entity": entity, "value": ""})
        return final_entities

    def _parse_slot(self, step: Dict[Text, Any]) -> None:

        slot_name = step.get(KEY_SLOT_NAME, "")
        slot_value = step.get(KEY_SLOT_VALUE, "")

        if not slot_name or not slot_value:
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"Slots should have a name and a value. "
                "This story step will be skipped:\n"
                f"{step}",
                docs=DOCS_URL_STORIES,
            )
            return

        self._add_event(SlotSet.type_name, {slot_name: slot_value})

    def _parse_action(self, step: Dict[Text, Any]) -> None:

        action_name = step.get(KEY_ACTION, "")
        if not action_name:
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"Action name cannot be empty. "
                f"This story step will be skipped:\n"
                f"{step}",
                docs=DOCS_URL_STORIES,
            )
            return

        self._add_event(action_name, {})

    def _parse_bot_message(self, step: Dict[Text, Any]) -> None:
        bot_message = step.get(KEY_BOT_END_TO_END_MESSAGE, "")
        # TODO: Replace with schema validation
        if not bot_message:
            common_utils.raise_warning(
                f"Issue found in '{self.source_name}':\n"
                f"Bot message cannot be empty. "
                f"This story step will be skipped:\n"
                f"{step}",
                docs=DOCS_URL_STORIES,
            )
            return

        self._add_event(bot_message, {"e2e_text": bot_message})

    def _parse_checkpoint(self, step: Dict[Text, Any]) -> None:

        checkpoint_name = step.get(KEY_CHECKPOINT, "")
        slots = step.get(KEY_CHECKPOINT_SLOTS, [])

        slots_dict = {}

        for slot in slots:
            if not isinstance(slot, dict):
                common_utils.raise_warning(
                    f"Issue found in '{self.source_name}': \n"
                    f'Checkpoint "{checkpoint_name}" has a invalid slot: '
                    f"{slots}\n"
                    f"Items under the '{KEY_CHECKPOINT_SLOTS}' key must be YAML dictionaries. "
                    f"The checkpoint will be skipped.",
                    docs=DOCS_URL_STORIES,
                )
                return

            for key, value in slot.items():
                slots_dict[key] = value

        self._add_checkpoint(checkpoint_name, slots_dict)

    @staticmethod
    def is_yaml_story_file(file_path: Text) -> bool:
        """Checks if the specified file potentially contains
           Core training data in YAML format.

        Args:
            file_path: Path of the file to check.

        Returns:
            `True` in case the file is a Core YAML training data file,
            `False` otherwise.
        """
        suffix = Path(file_path).suffix

        if suffix and suffix not in YAML_FILE_EXTENSIONS:
            return False
        try:
            content = rasa.utils.io.read_yaml_file(file_path)
            if KEY_STORIES in content:
                return True
        except Exception as e:
            # Using broad `Exception` because yaml library is not exposing all Errors
            common_utils.raise_warning(
                f"Tried to check if '{file_path}' is a story file, but failed to "
                f"read it. If this file contains story data, you should "
                f"investigate this error, otherwise it is probably best to "
                f"move the file to a different location. "
                f"Error: {e}"
            )
            return False
