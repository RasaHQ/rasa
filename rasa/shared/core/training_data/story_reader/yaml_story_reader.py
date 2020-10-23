import logging
from pathlib import Path
from typing import Dict, Text, List, Any, Optional, Union

import rasa.shared.data
from rasa.shared.exceptions import YamlException
import rasa.shared.utils.io
from rasa.shared.core.constants import LOOP_NAME
from rasa.shared.nlu.constants import ENTITIES, INTENT_NAME_KEY
from rasa.shared.nlu.training_data import entities_parser
import rasa.shared.utils.validation

from rasa.shared.constants import (
    INTENT_MESSAGE_PREFIX,
    DOCS_URL_STORIES,
    TEST_STORIES_FILE_PREFIX,
    DOCS_URL_RULES,
)

from rasa.shared.core.constants import RULE_SNIPPET_ACTION_NAME
from rasa.shared.core.events import UserUttered, SlotSet, ActiveLoop
from rasa.shared.core.training_data.story_reader.story_reader import StoryReader
from rasa.shared.core.training_data.structures import StoryStep

logger = logging.getLogger(__name__)

KEY_STORIES = "stories"
KEY_STORY_NAME = "story"
KEY_RULES = "rules"
KEY_RULE_NAME = "rule"
KEY_STEPS = "steps"
KEY_ENTITIES = "entities"
KEY_USER_INTENT = "intent"
KEY_USER_MESSAGE = "user"
KEY_SLOT_NAME = "slot_was_set"
KEY_SLOT_VALUE = "value"
KEY_ACTIVE_LOOP = "active_loop"
KEY_ACTION = "action"
KEY_CHECKPOINT = "checkpoint"
KEY_CHECKPOINT_SLOTS = "slot_was_set"
KEY_METADATA = "metadata"
KEY_OR = "or"
KEY_RULE_CONDITION = "condition"
KEY_WAIT_FOR_USER_INPUT_AFTER_RULE = "wait_for_user_input"
KEY_RULE_FOR_CONVERSATION_START = "conversation_start"


CORE_SCHEMA_FILE = "utils/schemas/stories.yml"


class YAMLStoryReader(StoryReader):
    """Class that reads Core training data and rule data in YAML format."""

    @classmethod
    def from_reader(cls, reader: "YAMLStoryReader") -> "YAMLStoryReader":
        """Create a reader from another reader.

        Args:
            reader: Another reader.

        Returns:
            A new reader instance.
        """
        return cls(
            reader.domain,
            reader.template_variables,
            reader.use_e2e,
            reader.source_name,
            reader._is_used_for_training,
        )

    def read_from_file(self, filename: Union[Text, Path]) -> List[StoryStep]:
        """Read stories or rules from file.

        Args:
            filename: Path to the story/rule file.

        Returns:
            `StoryStep`s read from `filename`.
        """
        self.source_name = filename
        try:
            return self.read_from_string(
                rasa.shared.utils.io.read_file(
                    filename, rasa.shared.utils.io.DEFAULT_ENCODING
                )
            )
        except YamlException as e:
            e.filename = filename
            raise e

    def read_from_string(self, string: Text) -> List[StoryStep]:
        """Read stories or rules from a string.

        Args:
            string: Unprocessed YAML file content.

        Returns:
            `StoryStep`s read from `string`.
        """
        rasa.shared.utils.validation.validate_yaml_schema(string, CORE_SCHEMA_FILE)
        yaml_content = rasa.shared.utils.io.read_yaml(string)

        return self.read_from_parsed_yaml(yaml_content)

    def read_from_parsed_yaml(
        self, parsed_content: Dict[Text, Union[Dict, List]]
    ) -> List[StoryStep]:
        """Read stories from parsed YAML.

        Args:
            parsed_content: The parsed YAML as a dictionary.

        Returns:
            The parsed stories or rules.
        """

        if not rasa.shared.utils.validation.validate_training_data_format_version(
            parsed_content, self.source_name
        ):
            return []

        for key, parser_class in {
            KEY_STORIES: StoryParser,
            KEY_RULES: RuleParser,
        }.items():
            data = parsed_content.get(key) or []
            parser = parser_class.from_reader(self)
            parser.parse_data(data)
            self.story_steps.extend(parser.get_steps())

        return self.story_steps

    @classmethod
    def is_stories_file(cls, file_path: Text) -> bool:
        """Check if file contains Core training data or rule data in YAML format.

        Args:
            file_path: Path of the file to check.

        Returns:
            `True` in case the file is a Core YAML training data or rule data file,
            `False` otherwise.

        Raises:
            YamlException: if the file seems to be a YAML file (extension) but
                can not be read / parsed.
        """
        return rasa.shared.data.is_likely_yaml_file(file_path) and cls.is_key_in_yaml(
            file_path, KEY_STORIES, KEY_RULES
        )

    @classmethod
    def is_key_in_yaml(cls, file_path: Text, *keys: Text) -> bool:
        """Check if all keys are contained in the parsed dictionary from a yaml file.

        Arguments:
            file_path: path to the yaml file
            keys: keys to look for

        Returns:
              `True` if all the keys are contained in the file, `False` otherwise.

        Raises:
            YamlException: if the file seems to be a YAML file (extension) but
                can not be read / parsed.
        """
        content = rasa.shared.utils.io.read_yaml_file(file_path)
        return any(key in content for key in keys)

    @classmethod
    def _has_test_prefix(cls, file_path: Text) -> bool:
        """Check if the filename of a file at a path has a certain prefix.

        Arguments:
            file_path: path to the file

        Returns:
            `True` if the filename starts with the prefix, `False` otherwise.
        """
        return Path(file_path).name.startswith(TEST_STORIES_FILE_PREFIX)

    @classmethod
    def is_test_stories_file(cls, file_path: Union[Text, Path]) -> bool:
        """Checks if a file is a test conversations file.

        Args:
            file_path: Path of the file which should be checked.

        Returns:
            `True` if it's a conversation test file, otherwise `False`.
        """

        return cls._has_test_prefix(file_path) and cls.is_stories_file(file_path)

    def get_steps(self) -> List[StoryStep]:
        self._add_current_stories_to_result()
        return self.story_steps

    def parse_data(self, data: List[Dict[Text, Any]]) -> None:
        item_title = self._get_item_title()

        for item in data:
            if not isinstance(item, dict):
                rasa.shared.utils.io.raise_warning(
                    f"Unexpected block found in '{self.source_name}':\n"
                    f"{item}\nItems under the "
                    f"'{self._get_plural_item_title()}' key must be YAML "
                    f"dictionaries. It will be skipped.",
                    docs=self._get_docs_link(),
                )
                continue

            if item_title in item.keys():
                self._parse_plain_item(item)

    def _parse_plain_item(self, item: Dict[Text, Any]) -> None:
        item_name = item.get(self._get_item_title(), "")

        if not item_name:
            rasa.shared.utils.io.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"{item}\n"
                f"The {self._get_item_title()} has an empty name. "
                f"{self._get_plural_item_title().capitalize()} should "
                f"have a name defined under '{self._get_item_title()}' "
                f"key. It will be skipped.",
                docs=self._get_docs_link(),
            )

        steps: List[Union[Text, Dict[Text, Any]]] = item.get(KEY_STEPS, [])

        if not steps:
            rasa.shared.utils.io.raise_warning(
                f"Issue found in '{self.source_name}': "
                f"The {self._get_item_title()} has no steps. "
                f"It will be skipped.",
                docs=self._get_docs_link(),
            )
            return

        self._new_part(item_name, item)

        for step in steps:
            self._parse_step(step)

        self._close_part(item)

    def _new_part(self, item_name: Text, item: Dict[Text, Any]) -> None:
        raise NotImplementedError()

    def _close_part(self, item: Dict[Text, Any]) -> None:
        pass

    def _parse_step(self, step: Union[Text, Dict[Text, Any]]) -> None:
        if isinstance(step, str):
            rasa.shared.utils.io.raise_warning(
                f"Issue found in '{self.source_name}':\n"
                f"Found an unexpected step in the {self._get_item_title()} "
                f"description:\n{step}\nThe step is of type `str` "
                f"which is only allowed for the rule snippet action "
                f"'{RULE_SNIPPET_ACTION_NAME}'. It will be skipped.",
                docs=self._get_docs_link(),
            )
        elif KEY_USER_INTENT in step.keys() or KEY_USER_MESSAGE in step.keys():
            self._parse_user_utterance(step)
        elif KEY_OR in step.keys():
            self._parse_or_statement(step)
        elif KEY_ACTION in step.keys():
            self._parse_action(step)
        elif KEY_CHECKPOINT in step.keys():
            self._parse_checkpoint(step)
        # This has to be after the checkpoint test as there can be a slot key within
        # a checkpoint.
        elif KEY_SLOT_NAME in step.keys():
            self._parse_slot(step)
        elif KEY_ACTIVE_LOOP in step.keys():
            self._parse_active_loop(step[KEY_ACTIVE_LOOP])
        elif KEY_METADATA in step.keys():
            pass
        else:
            rasa.shared.utils.io.raise_warning(
                f"Issue found in '{self.source_name}':\n"
                f"Found an unexpected step in the {self._get_item_title()} "
                f"description:\n{step}\nIt will be skipped.",
                docs=self._get_docs_link(),
            )

    def _get_item_title(self) -> Text:
        raise NotImplementedError()

    def _get_plural_item_title(self) -> Text:
        raise NotImplementedError()

    def _get_docs_link(self) -> Text:
        raise NotImplementedError()

    def _parse_user_utterance(self, step: Dict[Text, Any]) -> None:
        utterance = self._parse_raw_user_utterance(step)
        if utterance:
            self._validate_that_utterance_is_in_domain(utterance)
            self.current_step_builder.add_user_messages([utterance])

    def _validate_that_utterance_is_in_domain(self, utterance: UserUttered) -> None:
        intent_name = utterance.intent.get(INTENT_NAME_KEY)

        if not self.domain:
            logger.debug(
                "Skipped validating if intent is in domain as domain " "is `None`."
            )
            return

        if intent_name not in self.domain.intents:
            rasa.shared.utils.io.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"Found intent '{intent_name}' in stories which is not part of the "
                f"domain.",
                docs=DOCS_URL_STORIES,
            )

    def _parse_or_statement(self, step: Dict[Text, Any]) -> None:
        utterances = []

        for utterance in step.get(KEY_OR):
            if KEY_USER_INTENT in utterance.keys():
                utterance = self._parse_raw_user_utterance(utterance)
                if utterance:
                    utterances.append(utterance)
            else:
                rasa.shared.utils.io.raise_warning(
                    f"Issue found in '{self.source_name}': \n"
                    f"`OR` statement can only have '{KEY_USER_INTENT}' "
                    f"as a sub-element. This step will be skipped:\n"
                    f"'{utterance}'\n",
                    docs=self._get_docs_link(),
                )
                return

        self.current_step_builder.add_user_messages(
            utterances, self._is_used_for_training
        )

    def _user_intent_from_step(self, step: Dict[Text, Any]) -> Text:
        user_intent = step.get(KEY_USER_INTENT, "").strip()

        if not user_intent:
            rasa.shared.utils.io.raise_warning(
                f"Issue found in '{self.source_name}':\n"
                f"User utterance cannot be empty. "
                f"This {self._get_item_title()} step will be skipped:\n"
                f"{step}",
                docs=self._get_docs_link(),
            )

        if user_intent.startswith(INTENT_MESSAGE_PREFIX):
            rasa.shared.utils.io.raise_warning(
                f"Issue found in '{self.source_name}':\n"
                f"User intent '{user_intent}' starts with "
                f"'{INTENT_MESSAGE_PREFIX}'. This is not required.",
                docs=self._get_docs_link(),
            )
            # Remove leading slash
            user_intent = user_intent[1:]
        return user_intent

    def _parse_raw_user_utterance(self, step: Dict[Text, Any]) -> Optional[UserUttered]:
        from rasa.shared.nlu.interpreter import RegexInterpreter

        intent_name = self._user_intent_from_step(step)
        intent = {"name": intent_name, "confidence": 1.0}

        if KEY_USER_MESSAGE in step:
            user_message = step[KEY_USER_MESSAGE].strip()
            entities = entities_parser.find_entities_in_training_example(user_message)
            plain_text = entities_parser.replace_entities(user_message)

            if plain_text.startswith(INTENT_MESSAGE_PREFIX):
                entities = (
                    RegexInterpreter().synchronous_parse(plain_text).get(ENTITIES, [])
                )
        else:
            raw_entities = step.get(KEY_ENTITIES, [])
            entities = self._parse_raw_entities(raw_entities)
            # set plain_text to None because only intent was provided in the stories
            plain_text = None
        return UserUttered(plain_text, intent, entities)

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

        for slot in step.get(KEY_CHECKPOINT_SLOTS, []):
            if isinstance(slot, dict):
                for key, value in slot.items():
                    self._add_event(SlotSet.type_name, {key: value})
            elif isinstance(slot, str):
                self._add_event(SlotSet.type_name, {slot: None})
            else:
                rasa.shared.utils.io.raise_warning(
                    f"Issue found in '{self.source_name}':\n"
                    f"Invalid slot: \n{slot}\n"
                    f"Items under the '{KEY_CHECKPOINT_SLOTS}' key must be "
                    f"YAML dictionaries or Strings. The checkpoint will be skipped.",
                    docs=self._get_docs_link(),
                )
                return

    def _parse_action(self, step: Dict[Text, Any]) -> None:

        action_name = step.get(KEY_ACTION, "")
        if not action_name:
            rasa.shared.utils.io.raise_warning(
                f"Issue found in '{self.source_name}': \n"
                f"Action name cannot be empty. "
                f"This {self._get_item_title()} step will be skipped:\n"
                f"{step}",
                docs=self._get_docs_link(),
            )
            return

        self._add_event(action_name, {})

    def _parse_active_loop(self, active_loop_name: Optional[Text]) -> None:
        self._add_event(ActiveLoop.type_name, {LOOP_NAME: active_loop_name})

    def _parse_checkpoint(self, step: Dict[Text, Any]) -> None:

        checkpoint_name = step.get(KEY_CHECKPOINT, "")
        slots = step.get(KEY_CHECKPOINT_SLOTS, [])

        slots_dict = {}

        for slot in slots:
            if not isinstance(slot, dict):
                rasa.shared.utils.io.raise_warning(
                    f"Issue found in '{self.source_name}':\n"
                    f"Checkpoint '{checkpoint_name}' has an invalid slot: "
                    f"{slots}\nItems under the '{KEY_CHECKPOINT_SLOTS}' key must be "
                    f"YAML dictionaries. The checkpoint will be skipped.",
                    docs=self._get_docs_link(),
                )
                return

            for key, value in slot.items():
                slots_dict[key] = value

        self._add_checkpoint(checkpoint_name, slots_dict)


class StoryParser(YAMLStoryReader):
    """Encapsulate story-specific parser behavior."""

    def _new_part(self, item_name: Text, item: Dict[Text, Any]) -> None:
        self._new_story_part(item_name, self.source_name)

    def _get_item_title(self) -> Text:
        return KEY_STORY_NAME

    def _get_plural_item_title(self) -> Text:
        return KEY_STORIES

    def _get_docs_link(self) -> Text:
        return DOCS_URL_STORIES


class RuleParser(YAMLStoryReader):
    """Encapsulate rule-specific parser behavior."""

    def _new_part(self, item_name: Text, item: Dict[Text, Any]) -> None:
        self._new_rule_part(item_name, self.source_name)
        conditions = item.get(KEY_RULE_CONDITION, [])
        self._parse_rule_conditions(conditions)
        if not item.get(KEY_RULE_FOR_CONVERSATION_START):
            self._parse_rule_snippet_action()

    def _parse_rule_conditions(
        self, conditions: List[Union[Text, Dict[Text, Any]]]
    ) -> None:
        self._is_parsing_conditions = True
        for condition in conditions:
            self._parse_step(condition)
        self._is_parsing_conditions = False

    def _close_part(self, item: Dict[Text, Any]) -> None:
        if item.get(KEY_WAIT_FOR_USER_INPUT_AFTER_RULE) is False:
            self._parse_rule_snippet_action()

    def _get_item_title(self) -> Text:
        return KEY_RULE_NAME

    def _get_plural_item_title(self) -> Text:
        return KEY_RULES

    def _get_docs_link(self) -> Text:
        return DOCS_URL_RULES

    def _parse_rule_snippet_action(self) -> None:
        self._add_event(RULE_SNIPPET_ACTION_NAME, {})
