from __future__ import annotations

import logging
import typing
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Text

import rasa.utils.common
import rasa.utils.io

if typing.TYPE_CHECKING:
    from rasa.engine.storage.storage import ModelStorage

logger = logging.getLogger(__name__)


@dataclass
class Resource:
    """Represents a persisted graph component in the graph.

    Attributes:
        name: The unique identifier for the `Resource`. Used to locate the associated
            data from a `ModelStorage`. Normally matches the name of the node which
            created it.
        output_fingerprint: An unique identifier for a specific instantiation of a
            `Resource`. Used to distinguish a specific persistence for the same
            `Resource` when saving to the cache.

    """

    name: Text
    output_fingerprint: Text = field(
        default_factory=lambda: uuid.uuid4().hex,
        # We do not use this for comparison as it is not consistent after serialization.
        compare=False,
    )

    @classmethod
    def from_cache(
        cls,
        node_name: Text,
        directory: Path,
        model_storage: ModelStorage,
        output_fingerprint: Text,
    ) -> Resource:
        """Loads a `Resource` from the cache.

        This automatically loads the persisted resource into the given `ModelStorage`.

        Args:
            node_name: The node name of the `Resource`.
            directory: The directory with the cached `Resource`.
            model_storage: The `ModelStorage` which the cached `Resource` will be added
                to so that the `Resource` is accessible for other graph nodes.
            output_fingerprint: The fingerprint of the cached `Resource`.

        Returns:
            The ready-to-use and accessible `Resource`.
        """
        logger.debug(f"Loading resource '{node_name}' from cache.")

        resource = Resource(node_name, output_fingerprint=output_fingerprint)
        if not any(directory.glob("*")):
            logger.debug(f"Cached resource for '{node_name}' was empty.")
            return resource

        try:
            with model_storage.write_to(resource) as resource_directory:
                rasa.utils.common.copy_directory(directory, resource_directory)
        except ValueError:
            # This might happen during finetuning as in this case the model storage
            # is already filled
            if not rasa.utils.io.are_directories_equal(directory, resource_directory):
                # We skip caching in case we see the cached output and output
                # from the model which we want to finetune are not the same
                raise

        logger.debug(f"Successfully initialized resource '{node_name}' from cache.")

        return resource

    def to_cache(self, directory: Path, model_storage: ModelStorage) -> None:
        """Persists the `Resource` to the cache.

        Args:
            directory: The directory which receives the persisted `Resource`.
            model_storage: The model storage which currently contains the persisted
                `Resource`.
        """
        try:
            with model_storage.read_from(self) as resource_directory:
                rasa.utils.common.copy_directory(resource_directory, directory)
        except ValueError:
            logger.debug(
                f"Skipped caching resource '{self.name}' as no persisted "
                f"data was found."
            )

    def fingerprint(self) -> Text:
        """Provides fingerprint for `Resource`.

        A unique fingerprint is created on initialization of a `Resource` however we
        also allow a value to be provided for when we retrieve a `Resource` from the
        cache (see `Resource.from_cache`).

        Returns:
            Fingerprint for `Resource`.
        """
        return self.output_fingerprint
