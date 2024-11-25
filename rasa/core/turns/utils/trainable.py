from abc import ABC
from typing import Text

import rasa.shared.utils.io


class Trainable(ABC):
    def __init__(self) -> None:
        self._trained = False

    def mark_as_trained(self):
        self._trained = True

    def mark_as_not_trained(self):
        self._trained = False

    def __str__(self):
        return f"{self.__class__.__name__}(trained={self._trained})"

    def raise_if_not_trained(self, message: Text = "") -> None:
        if not self._trained:
            raise RuntimeError(
                f"Expected this {self.__class__.__name__} to be trained. " f"{message}"
            )

    def warn_if_not_trained(self, message: Text = "") -> None:
        if not self._trained:
            rasa.shared.utils.io.raise_warning(
                f"Expected this {self.__class__.__name__} to be trained. " f"{message}"
            )
