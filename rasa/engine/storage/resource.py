from __future__ import annotations
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Text

import rasa.utils.common

if typing.TYPE_CHECKING:
    from rasa.engine.storage.storage import ModelStorage

logger = logging.getLogger(__name__)


@dataclass
class Resource:
    """Represents a persisted graph component in the graph."""

    name: Text

    @classmethod
    def from_cache(
        cls, resource_name: Text, cache_directory: Path, model_storage: ModelStorage,
    ) -> Resource:
        """Loads a `Resource` from the cache.

        This automatically loads the persisted resource into the given `ModelStorage`.

        Args:
            resource_name: The name of the `Resource`.
            cache_directory: The directory with the cached `Resource`.
            model_storage: The `ModelStorage` which the cached `Resource` will be added
                to so that the `Resource` is accessible for other graph nodes.

        Returns:
            The ready-to-use and accessible `Resource`.
        """
        logger.debug(f"Loading resource '{resource_name}' from cache.")

        resource = Resource(resource_name)
        with model_storage.write_to(resource) as resource_directory:
            rasa.utils.common.copy_directory(cache_directory, resource_directory)

        logger.debug(f"Successfully initialized resource '{resource_name}' from cache.")

        return resource

    def to_cache(self, cache_directory: Path, model_storage: ModelStorage) -> None:
        """Persists the `Resource` to the cache.

        Args:
            cache_directory: The directory which receives the persisted `Resource`.
            model_storage: The model storage which currently contains the persisted
                `Resource`.
        """
        with model_storage.read_from(self) as resource_directory:
            rasa.utils.common.copy_directory(resource_directory, cache_directory)

    def fingerprint(self) -> Text:
        """Provides fingerprint for `Resource`.

        The fingerprint can be just the name as the persisted resource only changes
        if the used training data (which is loaded in previous nodes) or the config
        (which is fingerprinted separately) changes.

        Returns:
            Fingerprint for `Resource`.
        """
        return self.name
