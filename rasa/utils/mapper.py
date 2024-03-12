from pathlib import Path
from typing import Dict, List, Optional, Text, Union

import rasa.shared.data
import rasa.shared.utils.io
from rasa.shared.core.domain import (
    KEY_ACTIONS,
    KEY_ENTITIES,
    KEY_FORMS,
    KEY_INTENTS,
    KEY_RESPONSES,
    KEY_SLOTS,
    Domain,
)
from rasa.shared.core.flows.yaml_flows_io import KEY_FLOWS, is_flows_file
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    KEY_RULE_NAME,
    KEY_RULES,
    KEY_STORIES,
    KEY_STORY_NAME,
    YAMLStoryReader,
)
from rasa.shared.nlu.training_data.formats.rasa_yaml import (
    KEY_INTENT,
    KEY_NLU,
)
from rasa.shared.utils.yaml import read_yaml_file


class RasaPrimitiveStorageMapper:
    """Maps the Rasa primitives to the file its located in."""

    def __init__(
        self,
        domain_path: Optional[Union[Text, Path]] = None,
        training_data_paths: Optional[Union[List[Text], List[Path], Text, Path]] = None,
    ):
        self._primitives: Dict[str, Dict] = {
            "entities": {},
            "slots": {},
            "forms": {},
            "intents": {},
            "stories": {},
            "rules": {},
            "actions": {},
            "responses": {},
            "flows": {},
        }

        if domain_path:
            self._load_domain(domain_path)

        if training_data_paths:
            self._load_training_data(training_data_paths)

    def _load_domain(self, domain_path: Union[Text, Path]) -> None:
        self._resolve_domain_files(domain_path)
        self._parse_domain_files()

    def _load_training_data(
        self, training_data_paths: Union[List[Text], List[Path], Text, Path]
    ) -> None:
        self._resolve_training_data_files(training_data_paths)

        self._nlu_files = rasa.shared.data.get_data_files(
            self._training_data_paths, rasa.shared.data.is_nlu_file
        )
        self._parse_nlu_data()

        self._story_files = rasa.shared.data.get_data_files(
            self._training_data_paths, YAMLStoryReader.is_stories_file
        )
        self._parse_story_data()

        self._flow_files = rasa.shared.data.get_data_files(
            self._training_data_paths, is_flows_file
        )
        self._parse_flow_data()

    def _resolve_training_data_files(
        self, training_data_paths: Union[List[Text], List[Path], Text, Path]
    ) -> None:
        if isinstance(training_data_paths, list):
            training_data_paths = [Path(p) for p in training_data_paths]
        elif isinstance(training_data_paths, str):
            training_data_paths = [Path(training_data_paths)]
        elif isinstance(training_data_paths, Path):
            training_data_paths = [training_data_paths]

        self._training_data_paths = training_data_paths

    def _resolve_domain_files(self, domain_path: Union[Text, Path]) -> None:
        domain_path = Path(domain_path) if isinstance(domain_path, str) else domain_path

        if domain_path.is_dir():
            self._domain_files = [
                file for file in domain_path.iterdir() if Domain.is_domain_file(file)
            ]
        else:
            self._domain_files = [domain_path]

    def _parse_domain_files(self) -> None:
        for domain_file in self._domain_files:
            content = read_yaml_file(domain_file)
            for entity in content.get(KEY_ENTITIES, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["entities"], "domain", domain_file, entity
                )
            for intent in content.get(KEY_INTENTS, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["intents"], "domain", domain_file, intent
                )
            for slot in content.get(KEY_SLOTS, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["slots"], "domain", domain_file, slot
                )
            for form in content.get(KEY_FORMS, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["forms"], "domain", domain_file, form
                )
            for action in content.get(KEY_ACTIONS, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["actions"], "domain", domain_file, action
                )
            for response in content.get(KEY_RESPONSES, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["responses"], "domain", domain_file, response
                )

    def _parse_nlu_data(self) -> None:
        """Parses the nlu data and extracts the intents."""
        for nlu_file in self._nlu_files:
            content = read_yaml_file(nlu_file)
            for intent in content.get(KEY_NLU, []):  # type: ignore[union-attr]
                if KEY_INTENT in intent:
                    self._assign_value(
                        self._primitives["intents"],
                        "training",
                        nlu_file,
                        intent.get(KEY_INTENT),
                    )

    def _parse_story_data(self) -> None:
        """Parses the story data and extracts the stories and rules."""
        for story_file in self._story_files:
            content = read_yaml_file(story_file)
            for story in content.get(KEY_STORIES, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["stories"],
                    "training",
                    story_file,
                    story[KEY_STORY_NAME],
                )
            for rule in content.get(KEY_RULES, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["rules"],
                    "training",
                    story_file,
                    rule[KEY_RULE_NAME],
                )

    def _parse_flow_data(self) -> None:
        """Parses the flow data and extracts the flow ids."""
        for flow_file in self._flow_files:
            content = read_yaml_file(flow_file)
            for flow in content.get(KEY_FLOWS, []):  # type: ignore[union-attr]
                self._assign_value(
                    self._primitives["flows"],
                    "training",
                    flow_file,
                    flow,
                )

    @staticmethod
    def _assign_value(
        primitive: Dict, file_type: str, value: Union[str, Path], key: str
    ) -> None:
        if not isinstance(value, Path):
            value = Path(value)

        if isinstance(key, dict):
            key = next(iter(key.keys()))

        if key in primitive:
            if file_type in primitive[key]:
                primitive[key][file_type].append(value)
            else:
                primitive[key][file_type] = [value]
        else:
            primitive[key] = {file_type: [value]}

    def get_file(self, primitive: Text, primitive_type: Text) -> Dict[str, List[Path]]:
        """Returns the file where the primitive is located.

        If the primitive is not found, empty dict is returned.

        Args:
            primitive: The name or ID of the primitive to search for.
            primitive_type: The type of the primitive to search for.
                either entities, slots, forms, intents, stories or rules.

        Returns:
            A dictionary containing the file type and the list of file paths.
            file_type can be either domain or training.
            get_file()[file_type] -> List[Path]
        """
        try:
            return self._primitives[primitive_type][primitive]
        except KeyError:
            return {}
