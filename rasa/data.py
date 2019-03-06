import json
import os
import shutil
import tempfile
from typing import Tuple, List, Text
import re


def get_core_directory(directory: Text) -> Text:
    core_files, _ = _get_core_nlu_files(directory)
    return _copy_files_to_new_dir(core_files)


def get_nlu_directory(directory: Text) -> Text:
    _, nlu_files = _get_core_nlu_files(directory)
    return _copy_files_to_new_dir(nlu_files)


def get_core_nlu_directories(directory: Text) -> Tuple[Text, Text]:
    story_files, nlu_data_files = _get_core_nlu_files(directory)

    story_directory = _copy_files_to_new_dir(story_files)
    nlu_directory = _copy_files_to_new_dir(nlu_data_files)

    return story_directory, nlu_directory


def _get_core_nlu_files(directory: Text) -> Tuple[List[Text], List[Text]]:
    story_files = []
    nlu_data_files = []

    for root, _, files in os.walk(directory):
        for f in files:
            if not f.endswith(".json") and not f.endswith(".md"):
                continue

            full_path = os.path.join(root, f)
            if _is_nlu_file(full_path):
                nlu_data_files.append(full_path)
            else:
                story_files.append(full_path)

    return story_files, nlu_data_files


def _is_nlu_file(file_path: Text) -> bool:
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".json"):
            content = f.read()
            is_nlu_file = json.loads(content).get("rasa_nlu_data") is not None
        else:
            is_nlu_file = any(_contains_nlu_pattern(l) for l in f)
    return is_nlu_file


def _contains_nlu_pattern(text: Text) -> bool:
    nlu_pattern = r"\s*##\s*(intent|regex||synonym|lookup):"

    return re.match(nlu_pattern, text) is not None


def _copy_files_to_new_dir(files: List[Text]) -> Text:
    directory = tempfile.mkdtemp()
    for f in files:
        shutil.copy2(f, directory)

    return directory
