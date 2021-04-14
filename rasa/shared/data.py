import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Text, Optional, Union, List, Tuple, Callable, Set, Iterable
import rasa.shared.utils.io

YAML_FILE_EXTENSIONS = [".yml", ".yaml"]
JSON_FILE_EXTENSIONS = [".json"]
MARKDOWN_FILE_EXTENSIONS = [".md"]
TRAINING_DATA_EXTENSIONS = set(
    JSON_FILE_EXTENSIONS + MARKDOWN_FILE_EXTENSIONS + YAML_FILE_EXTENSIONS
)


def markdown_file_extension() -> Text:
    """Return Markdown file extension"""
    return MARKDOWN_FILE_EXTENSIONS[0]


def yaml_file_extension() -> Text:
    """Return Markdown file extension"""
    return YAML_FILE_EXTENSIONS[0]


def is_likely_yaml_file(file_path: Union[Text, Path]) -> bool:
    """Check if a file likely contains yaml.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in yaml format, `False` otherwise.
    """
    return Path(file_path).suffix in set(YAML_FILE_EXTENSIONS)


def is_likely_json_file(file_path: Text) -> bool:
    """Check if a file likely contains json.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in json format, `False` otherwise.
    """
    return Path(file_path).suffix in set(JSON_FILE_EXTENSIONS)


def is_likely_markdown_file(file_path: Text) -> bool:
    """Check if a file likely contains markdown.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in markdown format,
        `False` otherwise.
    """
    return Path(file_path).suffix in set(MARKDOWN_FILE_EXTENSIONS)


def get_test_directory(paths: Optional[Union[Text, List[Text]]]) -> Text:
    """Recursively collects all Core training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.

    Returns:
        Path to temporary directory containing all found Core training files.
    """
    rasa.shared.utils.io.raise_warning(
        "`get_test_directory` is deprecated and will be removed in Rasa 3.0.0. ",
        category=DeprecationWarning,
    )
    test_files = get_data_files(paths, is_test_stories_file)
    return _copy_files_to_new_dir(test_files)


def get_core_directory(paths: Optional[Union[Text, List[Text]]]) -> Text:
    """Recursively collects all Core training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.

    Returns:
        Path to temporary directory containing all found Core training files.
    """
    core_files = get_data_files(paths, is_story_file)
    return _copy_files_to_new_dir(core_files)


def get_nlu_directory(paths: Optional[Union[Text, List[Text]]],) -> Text:
    """Recursively collects all NLU training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.

    Returns:
        Path to temporary directory containing all found NLU training files.
    """
    nlu_files = get_data_files(paths, is_nlu_file)
    return _copy_files_to_new_dir(nlu_files)


def get_core_nlu_directories(
    paths: Optional[Union[Text, List[Text]]]
) -> Tuple[Text, Text]:
    """Recursively collects all training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.

    Returns:
        Path to directory containing the Core files and path to directory
        containing the NLU training files.
    """
    rasa.shared.utils.io.raise_warning(
        "`get_core_nlu_directories` is deprecated and will be removed in Rasa 3.0.0. ",
        category=DeprecationWarning,
    )
    story_files = get_data_files(paths, is_story_file)
    nlu_data_files = get_data_files(paths, is_nlu_file)

    story_directory = _copy_files_to_new_dir(story_files)
    nlu_directory = _copy_files_to_new_dir(nlu_data_files)

    return story_directory, nlu_directory


def get_data_files(
    paths: Optional[Union[Text, List[Text]]], filter_predicate: Callable[[Text], bool]
) -> List[Text]:
    """Recursively collects all training files from a list of paths.

    Args:
        paths: List of paths to training files or folders containing them.
        filter_predicate: property to use when filtering the paths, e.g. `is_nlu_file`.

    Returns:
        Paths of training data files.
    """

    data_files = set()

    if paths is None:
        paths = []
    elif isinstance(paths, str):
        paths = [paths]

    for path in set(paths):
        if not path:
            continue

        if is_valid_filetype(path):
            if filter_predicate(path):
                data_files.add(os.path.abspath(path))
        else:
            new_data_files = _find_data_files_in_directory(path, filter_predicate)
            data_files.update(new_data_files)

    return sorted(data_files)


def _find_data_files_in_directory(
    directory: Text, filter_property: Callable[[Text], bool]
) -> Set[Text]:
    filtered_files = set()

    for root, _, files in os.walk(directory, followlinks=True):
        # we sort the files here to ensure consistent order for repeatable training
        # results
        for f in sorted(files):
            full_path = os.path.join(root, f)

            if not is_valid_filetype(full_path):
                continue

            if filter_property(full_path):
                filtered_files.add(full_path)

    return filtered_files


def is_valid_filetype(path: Union[Path, Text]) -> bool:
    """Checks if given file has a supported extension.

    Args:
        path: Path to the source file.

    Returns:
        `True` is given file has supported extension, `False` otherwise.
    """
    return Path(path).is_file() and Path(path).suffix in TRAINING_DATA_EXTENSIONS


def is_nlu_file(file_path: Text) -> bool:
    """Checks if a file is a Rasa compatible nlu file.

    Args:
        file_path: Path of the file which should be checked.

    Returns:
        `True` if it's a nlu file, otherwise `False`.
    """
    from rasa.shared.nlu.training_data import loading as nlu_loading

    return nlu_loading.guess_format(file_path) != nlu_loading.UNK


def is_story_file(file_path: Text) -> bool:
    """Checks if a file is a Rasa story file.

    Args:
        file_path: Path of the file which should be checked.

    Returns:
        `True` if it's a story file, otherwise `False`.
    """
    from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
        YAMLStoryReader,
    )
    from rasa.shared.core.training_data.story_reader.markdown_story_reader import (
        MarkdownStoryReader,
    )

    return YAMLStoryReader.is_stories_file(
        file_path
    ) or MarkdownStoryReader.is_stories_file(file_path)


def is_test_stories_file(file_path: Text) -> bool:
    """Checks if a file is a test stories file.

    Args:
        file_path: Path of the file which should be checked.

    Returns:
        `True` if it's a story file containing tests, otherwise `False`.
    """
    from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
        YAMLStoryReader,
    )
    from rasa.shared.core.training_data.story_reader.markdown_story_reader import (
        MarkdownStoryReader,
    )

    return YAMLStoryReader.is_test_stories_file(
        file_path
    ) or MarkdownStoryReader.is_test_stories_file(file_path)


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
