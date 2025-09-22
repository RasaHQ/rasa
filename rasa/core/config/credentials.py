from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from rasa.shared.utils.yaml import read_config_file


class CredentialsConfig:
    def __init__(
        self, channels: Dict[str, Dict[str, Any]], config_file_path: Path
    ) -> None:
        self.channels = channels
        self.config_file_path = config_file_path

    @classmethod
    def load_from_file(cls, file_path: Path) -> CredentialsConfig:
        credentials_dict = read_config_file(file_path)
        return cls(credentials_dict, file_path)
