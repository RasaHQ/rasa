import logging
import os
import shutil
import tempfile
import uuid
import re
from typing import Tuple, List, Text, Set, Union, Optional, Iterable
from rasa.nlu.training_data import loading
from rasa.utils.io import DEFAULT_ENCODING

logger = logging.getLogger(__name__)


def get_core_directory(paths: Optional[Union[Text, List[Text]]],) -> Text:
    """Recursively collects all Core training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.

    Returns:
        Path to temporary directory containing all found Core training files.
    """
    core_files, _ = get_core_nlu_files(paths)
    return _copy_files_to_new_dir(core_files)


def get_nlu_directory(paths: Optional[Union[Text, List[Text]]],) -> Text:
    """Recursively collects all NLU training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.

    Returns:
        Path to temporary directory containing all found NLU training files.
    """
    _, nlu_files = get_core_nlu_files(paths)
    return _copy_files_to_new_dir(nlu_files)


def get_core_nlu_directories(
    paths: Optional[Union[Text, List[Text]]],
) -> Tuple[Text, Text]:
    """Recursively collects all training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.

    Returns:
        Path to directory containing the Core files and path to directory
        containing the NLU training files.
    """

    story_files, nlu_data_files = get_core_nlu_files(paths)

    story_directory = _copy_files_to_new_dir(story_files)
    nlu_directory = _copy_files_to_new_dir(nlu_data_files)

    return story_directory, nlu_directory


def get_core_nlu_files(
    paths: Optional[Union[Text, List[Text]]]
) -> Tuple[List[Text], List[Text]]:
    """Recursively collects all training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.

    Returns:
        Tuple of paths to story and NLU files.
    """

    story_files = set()
    nlu_data_files = set()

    if paths is None:
        paths = []
    elif isinstance(paths, str):
        paths = [paths]

    for path in set(paths):
        if not path:
            continue

        if _is_valid_filetype(path):
            if is_nlu_file(path):
                nlu_data_files.add(os.path.abspath(path))
            elif is_story_file(path):
                story_files.add(os.path.abspath(path))
        else:
            new_story_files, new_nlu_data_files = _find_core_nlu_files_in_directory(
                path
            )

            story_files.update(new_story_files)
            nlu_data_files.update(new_nlu_data_files)

    return sorted(story_files), sorted(nlu_data_files)


def _find_core_nlu_files_in_directory(directory: Text,) -> Tuple[Set[Text], Set[Text]]:
    story_files = set()
    nlu_data_files = set()

    for root, _, files in os.walk(directory, followlinks=True):
        # we sort the files here to ensure consistent order for repeatable training results
        for f in sorted(files):
            full_path = os.path.join(root, f)

            if not _is_valid_filetype(full_path):
                continue

            if is_nlu_file(full_path):
                nlu_data_files.add(full_path)
            elif is_story_file(full_path):
                story_files.add(full_path)

    return story_files, nlu_data_files


def _is_valid_filetype(path: Text) -> bool:
    is_file = os.path.isfile(path)
    is_datafile = path.endswith(".json") or path.endswith(".md")

    return is_file and is_datafile


def is_nlu_file(file_path: Text) -> bool:
    """Checks if a file is a Rasa compatible nlu file.

    Args:
        file_path: Path of the file which should be checked.

    Returns:
        `True` if it's a nlu file, otherwise `False`.
    """
    return loading.guess_format(file_path) != loading.UNK


def is_story_file(file_path: Text) -> bool:
    """Checks if a file is a Rasa story file.

    Args:
        file_path: Path of the file which should be checked.

    Returns:
        `True` if it's a story file, otherwise `False`.
    """

    if not file_path.endswith(".md"):
        return False

    try:
        with open(
            file_path, encoding=DEFAULT_ENCODING, errors="surrogateescape"
        ) as lines:
            return any(_contains_story_pattern(line) for line in lines)
    except Exception as e:
        # catch-all because we might be loading files we are not expecting to load
        logger.error(
            f"Tried to check if '{file_path}' is a story file, but failed to "
            f"read it. If this file contains story data, you should "
            f"investigate this error, otherwise it is probably best to "
            f"move the file to a different location. "
            f"Error: {e}"
        )
        return False


def _contains_story_pattern(text: Text) -> bool:
    story_pattern = r".*##.+"

    return re.match(story_pattern, text) is not None


def is_domain_file(file_path: Text) -> bool:
    """Checks whether the given file path is a Rasa domain file.

    Args:
        file_path: Path of the file which should be checked.

    Returns:
        `True` if it's a domain file, otherwise `False`.
    """

    file_name = os.path.basename(file_path)

    return file_name in ["domain.yml", "domain.yaml"]


def is_config_file(file_path: Text) -> bool:
    """Checks whether the given file path is a Rasa config file.

       Args:
           file_path: Path of the file which should be checked.

       Returns:
           `True` if it's a Rasa config file, otherwise `False`.
       """

    file_name = os.path.basename(file_path)

    return file_name in ["config.yml", "config.yaml"]


def _copy_files_to_new_dir(files: Iterable[Text]) -> Text:
    directory = tempfile.mkdtemp()
    for f in files:
        # makes sure files do not overwrite each other, hence the prefix
        unique_prefix = uuid.uuid4().hex
        unique_file_name = unique_prefix + "_" + os.path.basename(f)
        shutil.copy2(f, os.path.join(directory, unique_file_name))

    return directory
