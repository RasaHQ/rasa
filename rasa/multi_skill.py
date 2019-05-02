import logging
from typing import Text, List, Union, Set, Dict
import os

import rasa.utils.io as io_utils
from rasa import data

logger = logging.getLogger(__name__)


class SkillSelector:
    def __init__(self, imports: Set[Text], project_directory: Text = None):
        self.imports = imports
        self.project_directory = project_directory

    @classmethod
    def empty(cls) -> "SkillSelector":
        return cls(set())

    @classmethod
    def load(
        cls, config: Text, skill_paths: Union[Text, List[Text]]
    ) -> "SkillSelector":
        """
        Loads the specification from the config files.
        Args:
            config: Path to the root configuration file in the project directory.
            skill_paths: Paths which should be searched for further configuration files.

        Returns:
            `SkillSelector` which specifies the loaded skills.
        """
        # All imports are by default relative to the root config file directory
        config = os.path.abspath(config)
        selector = cls._from_file(config)

        if selector.is_empty():
            # if the root selector is empty we import everything beneath
            project_directory = os.path.dirname(config)
            selector.add_import(project_directory)

        if not isinstance(skill_paths, list):
            skill_paths = [skill_paths]

        for path in skill_paths:
            other = cls._from_path(path)
            selector = selector.merge(other)

        logger.debug("Selected skills: {}.".format(selector.imports))

        return selector

    @classmethod
    def _from_path(cls, path: Text) -> "SkillSelector":
        if os.path.isfile(path):
            return cls._from_file(path)
        elif os.path.isdir(path):
            return cls._from_directory(path)
        else:
            logger.debug("No imports found. Importing everything.")
            return cls.empty()

    @classmethod
    def _from_file(cls, path: Text) -> "SkillSelector":
        path = os.path.abspath(path)
        if data.is_config_file(path) and os.path.exists(path):
            config = io_utils.read_yaml_file(path)

            if isinstance(config, dict):
                parent_directory = os.path.dirname(path)
                return cls._from_dict(config, parent_directory)

        return cls.empty()

    @classmethod
    def _from_dict(cls, _dict: Dict, parent_directory: Text) -> "SkillSelector":
        imports = _dict.get("imports") or []
        imports = {os.path.join(parent_directory, i) for i in imports}
        # clean out relative paths
        imports = {os.path.abspath(i) for i in imports}

        return cls(imports, parent_directory)

    @classmethod
    def _from_directory(cls, path: Text) -> "SkillSelector":
        importer = cls.empty()
        for parent, _, files in os.walk(path):
            for file in files:
                full_path = os.path.join(parent, file)

                if data.is_config_file(full_path) and importer.is_imported(full_path):
                    other = cls._from_file(full_path)
                    importer = importer.merge(other)

        return importer

    def merge(self, other: "SkillSelector") -> "SkillSelector":
        self.imports |= {
            i for i in other.imports if not self.is_imported(i) or self.is_empty()
        }

        return self

    def is_empty(self) -> bool:
        return not self.imports

    def training_paths(self) -> Set[Text]:
        """Returns the paths which should be searched for training data."""

        # only include extra paths if they are not part of the current project directory
        training_paths = {i for i in self.imports if self.project_directory not in i}
        return training_paths | {self.project_directory}

    def is_imported(self, path: Text) -> bool:
        """
        Checks whether a path is imported by a skill.
        Args:
            path: File or directory path which should be checked.

        Returns:
            `True` if path is imported by a skill, `False` if not.
        """
        absolute_path = os.path.abspath(path)

        return (
            self.is_empty()
            or os.path.abspath(path) == self.project_directory
            or (
                os.path.isfile(absolute_path)
                and os.path.abspath(os.path.dirname(path)) == self.project_directory
            )
            or any([i in absolute_path for i in self.imports])
        )

    def add_import(self, path: Text) -> bool:
        self.imports.add(path)
