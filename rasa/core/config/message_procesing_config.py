from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from rasa.shared.utils.yaml import read_config_file


class MessageProcessingConfig:
    def __init__(
        self,
        recipe: str,
        language: str,
        assistant_id: str,
        pipeline: List[Dict[str, Any]],
        policies: List[Dict[str, Any]],
    ):
        self.recipe = recipe
        self.language = language
        self.assistant_id = assistant_id
        self.pipeline = pipeline
        self.policies = policies

    @classmethod
    def load_from_file(cls, filepath: Path) -> MessageProcessingConfig:
        credentials_dict = read_config_file(filepath)

        recipe = credentials_dict.get("recipe", "")
        language = credentials_dict.get("language", "")
        assistant_id = credentials_dict.get("assistant_id", "")
        pipeline = credentials_dict.get("pipeline", [])
        policies = credentials_dict.get("policies", [])

        return cls(recipe, language, assistant_id, pipeline, policies)
