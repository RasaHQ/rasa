import copy
from typing import Text, Dict, List
import functools
from rasa.shared.core.training_data.story_reader.yaml_story_reader import \
    CORE_SCHEMA_FILE, YAMLStoryReader, KEY_STORIES, KEY_RULES, KEY_STORY_NAME, \
    KEY_STEPS, KEY_USER_INTENT, KEY_ACTION, KEY_ENTITIES, KEY_SLOT_NAME, \
    KEY_ACTIVE_LOOP, KEY_CHECKPOINT, KEY_OR, KEY_RULE_NAME, KEY_RULE_CONDITION
import rasa.shared.utils.io
import rasa.shared.utils.validation
import rasa.shared.data
import rasa.shared.spaces.utils as space_utils
from rasa.shared.spaces.domain_resolver import DomainInfo


class StoryResolver:

    @classmethod
    def resolve_rules(cls, stories_yaml: Dict, prefix: Text,
                      domain_info: DomainInfo) -> None:
        """Resolve rules of a stories yaml dict."""
        for rule in stories_yaml[KEY_RULES]:
            # prefix rule name for debugging purposes
            rule[KEY_RULE_NAME] = f"({prefix}) {rule[KEY_RULE_NAME]}"
            for step in rule.get(KEY_STEPS, []):
                cls.resolve_step(step, prefix, domain_info)
            for condition in rule.get(KEY_RULE_CONDITION, []):
                cls.resolve_step(condition, prefix, domain_info)

    @classmethod
    def resolve_stories(cls, stories_yaml: Dict, prefix: Text,
                        domain_info: DomainInfo) -> None:
        """Resolve stories of a stories yaml dict."""
        for story in stories_yaml[KEY_STORIES]:
            # prefix story name for debugging purposes
            story[KEY_STORY_NAME] = f"({prefix}) {story[KEY_STORY_NAME]}"
            for step in story.get(KEY_STEPS, []):
                cls.resolve_step(step, prefix, domain_info)

    @classmethod
    def resolve_step(cls, step: Dict, prefix: Text,
                     domain_info: DomainInfo) -> None:
        """Resolve a single step of a rule or story."""

        if KEY_USER_INTENT in step:
            cls.resolve_user_step(step, prefix, domain_info)
        if KEY_ACTION in step:
            cls.resolve_bot_action(step, prefix, domain_info)
        if KEY_SLOT_NAME in step:
            cls.resolve_slot_set(step, prefix, domain_info)
        if step.get(KEY_ACTIVE_LOOP) in domain_info.forms:
            space_utils.prefix_dict_value(step, KEY_ACTIVE_LOOP, prefix)
        if KEY_CHECKPOINT in step:
            space_utils.prefix_dict_value(step, KEY_CHECKPOINT, prefix)
        if KEY_OR in step:
            for alternative_step in step[KEY_OR]:
                cls.resolve_step(alternative_step, prefix, domain_info)

    @classmethod
    def resolve_bot_action(cls, step: Dict, prefix: Text,
                           domain_info: DomainInfo) -> None:
        """Resolve a bot action in a rule or story."""
        potential_references = [domain_info.actions, domain_info.forms,
                                domain_info.responses]
        potential_references = functools.reduce(lambda a, b: a.union(b),
                                                potential_references, set())
        if step[KEY_ACTION] in potential_references:
            space_utils.prefix_dict_value_with_potential_utter(step, KEY_ACTION, prefix)

    @classmethod
    def resolve_user_step(cls, step: Dict, prefix: Text,
                          domain_info: DomainInfo) -> None:
        """Resolve a user step in a rule or story"""
        if step.get(KEY_USER_INTENT) in domain_info.intents:
            space_utils.prefix_dict_value(step, KEY_USER_INTENT, prefix)
        for i, entity in enumerate(step.get(KEY_ENTITIES, [])):
            entity_copy = copy.copy(entity)
            for key, value in entity.items():
                if key not in {"role", "group"} and key in domain_info.entities:
                    space_utils.prefix_dict_key(entity_copy, key, prefix)
            step[KEY_ENTITIES][i] = entity_copy

    @classmethod
    def resolve_slot_set(cls, step: Dict, prefix: Text,
                         domain_info: DomainInfo) -> None:
        for i, slot in enumerate(step.get(KEY_SLOT_NAME, [])):
            if isinstance(slot, dict):
                slot_copy = copy.copy(slot)
                for key in slot:
                    if key in domain_info.slots:
                        space_utils.prefix_dict_key(slot_copy, key, prefix)
                step[KEY_SLOT_NAME][i] = slot_copy
            elif isinstance(slot, str):
                if slot in domain_info.slots:
                    step[KEY_SLOT_NAME][i] = space_utils.prefix_string(slot, prefix)


    @classmethod
    def join_story_yaml_dicts(cls, story_yamls: List[Dict]) -> Dict:
        """Concatenate stories and rules of multiple yaml dicts."""
        joined_yaml_dict = {KEY_STORIES: [], KEY_RULES: []}
        for story_yaml in story_yamls:
            joined_yaml_dict[KEY_STORIES].extend(story_yaml.get(KEY_STORIES, []))
            joined_yaml_dict[KEY_RULES].extend(story_yaml.get(KEY_RULES, []))
        return joined_yaml_dict

    @classmethod
    def load(cls, stories_path: Text) -> Dict:
        """Load one or multiple story files into a joined yaml dict."""
        story_files = rasa.shared.data.get_data_files(
            stories_path, YAMLStoryReader.is_stories_file
        )
        story_yamls = []
        for story_file in story_files:
            yaml_string = rasa.shared.utils.io.read_file(story_file)
            rasa.shared.utils.validation.validate_yaml_schema(yaml_string,
                                                              CORE_SCHEMA_FILE)
            story_yamls.append(rasa.shared.utils.io.read_yaml(yaml_string))
        joined_yaml_dict = cls.join_story_yaml_dicts(story_yamls)
        return joined_yaml_dict

    @classmethod
    def load_and_resolve(cls, stories_path: Text, prefix: Text,
                         domain_info: DomainInfo) -> Dict:
        stories_yaml = cls.load(stories_path)
        cls.resolve_stories(stories_yaml, prefix, domain_info)
        cls.resolve_rules(stories_yaml, prefix, domain_info)
        return stories_yaml
