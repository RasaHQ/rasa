import logging
import os
import shutil
import tempfile
import uuid
from typing import Tuple, List, Text, Set, Union, Optional
import re

import rasa.utils.io as io_utils
from rasa.skill import SkillSelector

logger = logging.getLogger(__name__)


def get_core_directory(
    paths: Optional[Union[Text, List[Text]]],
    skill_imports: Optional[SkillSelector] = None,
) -> Text:
    """Recursively collects all Core training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.
        skill_imports: `SkillSelector` instance which determines which files should
                       be loaded.

    Returns:
        Path to temporary directory containing all found Core training files.
    """
    core_files, _ = get_core_nlu_files(paths, skill_imports)
    return _copy_files_to_new_dir(core_files)


def get_nlu_directory(
    paths: Optional[Union[Text, List[Text]]],
    skill_imports: Optional[SkillSelector] = None,
) -> Text:
    """Recursively collects all NLU training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.
        skill_imports: `SkillSelector` instance which determines which files should
                       be loaded.

    Returns:
        Path to temporary directory containing all found NLU training files.
    """
    _, nlu_files = get_core_nlu_files(paths, skill_imports)
    return _copy_files_to_new_dir(nlu_files)


def get_core_nlu_directories(
    paths: Optional[Union[Text, List[Text]]],
    skill_imports: Optional[SkillSelector] = None,
) -> Tuple[Text, Text]:
    """Recursively collects all training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.
        skill_imports: `SkillSelector` instance which determines which files should
                       be loaded.

    Returns:
        Path to directory containing the Core files and path to directory
        containing the NLU training files.
    """

    story_files, nlu_data_files = get_core_nlu_files(paths, skill_imports)

    story_directory = _copy_files_to_new_dir(story_files)
    nlu_directory = _copy_files_to_new_dir(nlu_data_files)

    return story_directory, nlu_directory


def get_core_nlu_files(
    paths: Optional[Union[Text, List[Text]]],
    skill_imports: Optional[SkillSelector] = None,
) -> Tuple[Set[Text], Set[Text]]:
    """Recursively collects all training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.
        skill_imports: `SkillSelector` instance which determines which files should
                       be loaded.

    Returns:
        Tuple of paths to story and NLU files.
    """

    story_files = set()
    nlu_data_files = set()

    skill_imports = skill_imports or SkillSelector.all_skills()

    if not skill_imports.no_skills_selected():
        paths = skill_imports.training_paths()

    if paths is None:
        paths = []
    elif isinstance(paths, str):
        paths = [paths]

    for path in set(paths):
        if not path:
            continue

        if _is_valid_filetype(path) and skill_imports.is_imported(path):
            if _is_nlu_file(path):
                nlu_data_files.add(os.path.abspath(path))
            elif _is_story_file(path):
                story_files.add(os.path.abspath(path))
        else:
            new_story_files, new_nlu_data_files = _find_core_nlu_files_in_directory(
                path, skill_imports
            )

            story_files.update(new_story_files)
            nlu_data_files.update(new_nlu_data_files)

    return story_files, nlu_data_files


def _find_core_nlu_files_in_directory(
    directory: Text, skill_imports: SkillSelector
) -> Tuple[Set[Text], Set[Text]]:
    story_files = set()
    nlu_data_files = set()

    for root, _, files in os.walk(directory):
        if not skill_imports.is_imported(root):
            continue

        for f in files:
            full_path = os.path.join(root, f)

            if not _is_valid_filetype(full_path):
                continue

            if _is_nlu_file(full_path):
                nlu_data_files.add(full_path)
            elif _is_story_file(full_path):
                story_files.add(full_path)

    return story_files, nlu_data_files


def _is_valid_filetype(path: Text) -> bool:
    is_file = os.path.isfile(path)
    is_datafile = path.endswith(".json") or path.endswith(".md")

    return is_file and is_datafile


def _is_nlu_file(file_path: Text) -> bool:
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".json"):
            content = io_utils.read_json_file(file_path)
            is_nlu_file = (
                isinstance(content, dict) and content.get("rasa_nlu_data") is not None
            )
        else:
            is_nlu_file = any(_contains_nlu_pattern(l) for l in f)
    return is_nlu_file


def _contains_nlu_pattern(text: Text) -> bool:
    nlu_pattern = r"\s*##\s*(intent|regex||synonym|lookup):"

    return re.match(nlu_pattern, text) is not None


def _is_story_file(file_path: Text) -> bool:
    is_story_file = False

    if file_path.endswith(".md"):
        with open(file_path, encoding="utf-8") as f:
            is_story_file = any(_contains_story_pattern(l) for l in f)

    return is_story_file


def _contains_story_pattern(text: Text) -> bool:
    story_pattern = r".*##.+"

    return re.match(story_pattern, text) is not None


def is_domain_file(file_path: Text) -> bool:
    """Checks whether the given file path is a Rasa domain file."""

    file_name = os.path.basename(file_path)

    return file_name in ["domain.yml", "domain.yaml"]


def is_config_file(file_path: Text) -> bool:
    """Checks whether the given file path is a Rasa config file."""

    file_name = os.path.basename(file_path)

    return file_name in ["config.yml", "config.yaml"]


def _copy_files_to_new_dir(files: Set[Text]) -> Text:
    directory = tempfile.mkdtemp()
    for f in files:
        # makes sure files do not overwrite each other, hence the prefix
        unique_prefix = uuid.uuid4().hex
        unique_file_name = unique_prefix + "_" + os.path.basename(f)
        shutil.copy2(f, os.path.join(directory, unique_file_name))

    return directory
