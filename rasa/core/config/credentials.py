from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from rasa.shared.utils.yaml import read_config_file


class CredentialsConfig:
    def __init__(
        self, channels: Dict[str, Dict[str, Any]], config_file_path: Path
    ) -> None:
        self.channels = channels
        self.config_file_path = config_file_path

        for _, config in self.channels.items():
            if config and "silence_timeout" in config:
                config["silence_timeout"] = self.validate_silence_timeout(
                    config["silence_timeout"]
                )

    @classmethod
    def load_from_file(cls, file_path: Path) -> CredentialsConfig:
        credentials_dict = read_config_file(file_path)
        return cls(credentials_dict, file_path)

    @classmethod
    def validate_silence_timeout(
        cls, silence_timeout: Union[float, int]
    ) -> Union[float, int]:
        """Validate the silence timeout value."""
        if isinstance(silence_timeout, str):
            try:
                silence_timeout = float(silence_timeout)
            except ValueError:
                raise ValueError(
                    f"Type error for silence timeout value: {silence_timeout}. "
                    "Silence timeout must be a positive number."
                )

        if not isinstance(silence_timeout, (float, int)):
            raise ValueError(
                f"Type error for silence timeout value: {silence_timeout}. "
                "Silence timeout must be a positive number."
            )

        if silence_timeout <= 0:
            raise ValueError(
                f"Value error for silence timeout value: {silence_timeout}. "
                "Silence timeout must be a positive number."
            )

        return silence_timeout
