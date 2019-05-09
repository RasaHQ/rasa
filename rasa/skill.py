import logging
from typing import Text, Set, Dict
import os

import rasa.utils.io as io_utils
from rasa import data

logger = logging.getLogger(__name__)


class SkillSelector:
    def __init__(self, imports: Set[Text], project_directory: Text = os.getcwd()):
        self._imports = imports
        self._project_directory = project_directory

    @classmethod
    def all_skills(cls, project_directory: Text = os.getcwd()) -> "SkillSelector":
        """Returns a `SkillSelector` instance which does not specify any skills."""

        return cls(set(), project_directory)

    @classmethod
    def load(cls, config: Text) -> "SkillSelector":
        """
        Loads the specification from the config files.
        Args:
            config: Path to the root configuration file in the project directory.

        Returns:
            `SkillSelector` which specifies the loaded skills.
        """
        # All imports are by default relative to the root config file directory
        config = os.path.abspath(config)

        # Create a base selector which keeps track of the imports during the
        # skill config loading in order to avoid cyclic imports
        selector = cls.all_skills(os.path.dirname(config))

        selector = cls._from_file(config, selector)

        logger.info("Selected skills: {}.".format(selector._imports))

        return selector

    @classmethod
    def _from_path(cls, path: Text, skill_selector: "SkillSelector") -> "SkillSelector":
        if os.path.isfile(path):
            return cls._from_file(path, skill_selector)
        elif os.path.isdir(path):
            return cls._from_directory(path, skill_selector)
        else:
            logger.debug("No imports found. Importing everything.")
            return cls.all_skills()

    @classmethod
    def _from_file(cls, path: Text, skill_selector: "SkillSelector") -> "SkillSelector":
        path = os.path.abspath(path)
        if os.path.exists(path) and data.is_config_file(path):
            config = io_utils.read_yaml_file(path)

            if isinstance(config, dict):
                parent_directory = os.path.dirname(path)
                return cls._from_dict(config, parent_directory, skill_selector)

        return cls.all_skills()

    @classmethod
    def _from_dict(
        cls, _dict: Dict, parent_directory: Text, skill_selector: "SkillSelector"
    ) -> "SkillSelector":
        imports = _dict.get("imports") or []
        imports = {os.path.join(parent_directory, i) for i in imports}
        # clean out relative paths
        imports = {os.path.abspath(i) for i in imports}
        import_candidates = [
            p for p in imports if not skill_selector._is_explicitly_imported(p)
        ]
        new = cls(imports, parent_directory)
        skill_selector = skill_selector.merge(new)

        # import config files from paths which have not been processed so far
        for p in import_candidates:
            other = cls._from_path(p, skill_selector)
            skill_selector = skill_selector.merge(other)

        return skill_selector

    def _is_explicitly_imported(self, path: Text) -> bool:
        return not self.no_skills_selected() and self.is_imported(path)

    @classmethod
    def _from_directory(
        cls, path: Text, skill_selector: "SkillSelector"
    ) -> "SkillSelector":
        for parent, _, files in os.walk(path):
            for file in files:
                full_path = os.path.join(parent, file)

                if data.is_config_file(full_path) and skill_selector.is_imported(
                    full_path
                ):
                    skill_selector = cls._from_file(full_path, skill_selector)

        return skill_selector

    def merge(self, other: "SkillSelector") -> "SkillSelector":
        imports = self._imports.union(
            {
                i
                for i in other._imports
                if not self.is_imported(i) or self.no_skills_selected()
            }
        )

        return SkillSelector(imports, self._project_directory)

    def no_skills_selected(self) -> bool:
        return not self._imports

    def training_paths(self) -> Set[Text]:
        """Returns the paths which should be searched for training data."""

        # only include extra paths if they are not part of the current project directory
        training_paths = {
            i
            for i in self._imports
            if not self._project_directory or self._project_directory not in i
        }

        if self._project_directory:
            training_paths.add(self._project_directory)

        return training_paths

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
            self.no_skills_selected()
            or self._is_in_project_directory(absolute_path)
            or self._is_in_imported_paths(absolute_path)
        )

    def _is_in_project_directory(self, path: Text) -> bool:
        if os.path.isfile(path):
            parent_directory = os.path.abspath(os.path.dirname(path))

            return parent_directory == self._project_directory
        else:
            return path == self._project_directory

    def _is_in_imported_paths(self, path):
        return any([io_utils.is_subdirectory(path, i) for i in self._imports])

    def add_import(self, path: Text) -> bool:
        self._imports.add(path)
