from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from rasa.shared.utils.yaml import read_config_file


class MessageProcessingConfig:
    def __init__(
        self,
        recipe: str,
        language: str,
        additional_languages: List[str],
        assistant_id: str,
        pipeline: List[Dict[str, Any]],
        policies: List[Dict[str, Any]],
    ):
        self.recipe = recipe
        self.language = language
        self.additional_languages = additional_languages
        self.assistant_id = assistant_id
        self.pipeline = pipeline
        self.policies = policies

    @classmethod
    def load_from_file(cls, filepath: Path) -> MessageProcessingConfig:
        credentials_dict = read_config_file(filepath)

        recipe = credentials_dict.get("recipe", "")
        language = credentials_dict.get("language", "")
        additional_languages = credentials_dict.get("additional_languages", [])
        assistant_id = credentials_dict.get("assistant_id", "")
        pipeline = credentials_dict.get("pipeline", [])
        policies = credentials_dict.get("policies", [])

        return cls(
            recipe, language, additional_languages, assistant_id, pipeline, policies
        )
