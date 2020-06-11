import logging
from typing import Optional, Dict, Text, List, Any

import rasa.utils.io
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.core.domain import Domain
from rasa.core.events import UserUttered, SlotSet
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.training.story_reader.story_reader import StoryReader
from rasa.core.training.structures import StoryStep

logger = logging.getLogger(__name__)

KEY_STORIES = "stories"
KEY_PLAIN_STORY = "story"
KEY_PLAIN_STORY_STEPS = "steps"
KEY_PLAIN_STORY_USER_UTTERANCE = "user"
KEY_SLOT_NAME = "slot"
KEY_SLOT_VALUE = "value"
KEY_ACTION = "action"
KEY_PLAIN_STORY_ENTITIES = "entities"


class YAMLStoryReader(StoryReader):
    def __init__(
        self,
        interpreter: NaturalLanguageInterpreter,
        domain: Optional[Domain] = None,
        template_vars: Optional[Dict] = None,
        use_e2e: bool = False,
        source_name: Text = None,
    ):
        super().__init__(interpreter, domain, template_vars, use_e2e, source_name)

    async def read_from_file(self, filename: Text) -> List[StoryStep]:
        try:
            file_content = rasa.utils.io.read_file(filename)
            yaml_content = rasa.utils.io.read_yaml(file_content)

            for key, value in yaml_content.items():
                if key == KEY_STORIES:
                    self._parse_stories(value)
                    self._add_current_stories_to_result()
                    return self.story_steps
                else:
                    logger.warning(f"Unexpected key {key} found in {self.source_name}")

        except ValueError:
            logger.error(f"Failed to read {filename}, it will be skipped")

        return []

    def _parse_stories(self, stories: List[Dict[Text, Any]]) -> None:

        for story_item in stories:
            if not isinstance(story_item, dict):
                logger.warning(
                    f"Unexpected block found in {self.source_name}: \n"
                    f"{story_item}\n"
                    f"Items under the `{KEY_STORIES}` key must be YAML dictionaries."
                    f"It will be skipped."
                )
                continue

            if KEY_PLAIN_STORY in story_item.keys():
                self._parse_plain_story(story_item)

    def _parse_plain_story(self, story_item: Dict[Text, Any]) -> None:
        story_name = story_item.get(KEY_PLAIN_STORY, "")
        steps = story_item.get(KEY_PLAIN_STORY_STEPS, "")

        self._new_story_part(story_name, self.source_name)

        for step in steps:
            self._parse_step(step)

    def _parse_step(self, step: Dict[Text, Any]):

        if KEY_PLAIN_STORY_USER_UTTERANCE in step.keys():
            self._parse_plain_story_user_utterance(step)
        if KEY_SLOT_NAME in step.keys():
            self._parse_slot(step)
        if KEY_ACTION in step.keys():
            self._parse_action(step)

    def _parse_plain_story_user_utterance(self, step: Dict[Text, Any]):

        user_utterance = step.get(KEY_PLAIN_STORY_USER_UTTERANCE, "").strip()
        entities = step.get(KEY_PLAIN_STORY_ENTITIES, "")

        if not user_utterance.startswith(INTENT_MESSAGE_PREFIX):
            logger.warning(
                f'User intent "{user_utterance}" should start with '
                f'"{INTENT_MESSAGE_PREFIX}"'
            )
        else:
            user_utterance = user_utterance[1:]

        entities_list = []
        for entity in entities:
            if isinstance(entity, dict):
                for key, value in entity.items():
                    entities_list.append({"entity": key, "value": value})
            else:
                entities_list.append({"entity": entity, "value": ""})

        intent = {"name": user_utterance, "confidence": 1.0}
        utterance = UserUttered(user_utterance, intent, entities_list)

        if self.use_e2e:
            # TODO
            pass
        else:
            self.current_step_builder.add_user_messages([utterance])

    def _parse_slot(self, step: Dict[Text, Any]) -> None:

        slot_name = step.get(KEY_SLOT_NAME, "")
        slot_value = step.get(KEY_SLOT_VALUE, "")

        self._add_event(SlotSet.type_name, {slot_name: slot_value})

    def _parse_action(self, step: Dict[Text, Any]) -> None:

        action_name = step.get(KEY_ACTION, "")
        self._add_event(action_name, {})
