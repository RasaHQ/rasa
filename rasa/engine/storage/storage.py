from __future__ import annotations
import abc
import logging
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, Text, ContextManager, Dict, Any

from rasa.engine.storage.resource import Resource
from rasa.shared.core.domain import Domain
from rasa.shared.importers.autoconfig import TrainingType

if typing.TYPE_CHECKING:
    from rasa.engine.graph import GraphSchema


logger = logging.getLogger(__name__)


class ModelStorage(abc.ABC):
    """Serves as storage backend for `GraphComponents` which need persistence."""

    @classmethod
    @abc.abstractmethod
    def create(cls, storage_path: Path) -> ModelStorage:
        """Creates the storage.

        Args:
            storage_path: Directory which will contain the persisted graph components.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def from_model_archive(
        cls, storage_path: Path, model_archive_path: Union[Text, Path]
    ) -> Tuple[ModelStorage, ModelMetadata]:
        """Unpacks a model archive and initializes a `ModelStorage`.

        Args:
            storage_path: Directory which will contain the persisted graph components.
            model_archive_path: The path to the model archive.

        Returns:
            Initialized model storage, and metadata about the model.
        """
        ...

    @classmethod
    def metadata_from_archive(
        cls, model_archive_path: Union[Text, Path]
    ) -> ModelMetadata:
        """Retrieve metadata from archive.

        Args:
            model_archive_path: The path to the model archive.

        Returns:
            Metadata about the model.
        """
        ...

    @contextmanager
    @abc.abstractmethod
    def write_to(self, resource: Resource) -> ContextManager[Path]:
        """Persists data for a given resource.

        This `Resource` can then be accessed in dependent graph nodes via
        `model_storage.read_from`.

        Args:
            resource: The resource which should be persisted.

        Returns:
            A directory which can be used to persist data for the given `Resource`.
        """
        ...

    @contextmanager
    @abc.abstractmethod
    def read_from(self, resource: Resource) -> ContextManager[Path]:
        """Provides the data of a persisted `Resource`.

        Args:
            resource: The `Resource` whose persisted should be accessed.

        Returns:
            A directory containing the data of the persisted `Resource`.

        Raises:
            ValueError: In case no persisted data for the given `Resource` exists.
        """
        ...

    def create_model_package(
        self,
        model_archive_path: Union[Text, Path],
        train_schema: GraphSchema,
        predict_schema: GraphSchema,
        domain: Domain,
        training_type: TrainingType = TrainingType.BOTH,
    ) -> ModelMetadata:
        """Creates a model archive containing all data to load and run the model.

        Args:
            model_archive_path: The path to the archive which should be created.
            train_schema: The schema which was used to train the graph model.
            predict_schema: The schema for running predictions with the trained model.
            domain: The `Domain` which was used to train the model.
            training_type: NLU, CORE or BOTH depending on what is trained.

        Returns:
            The model metadata.
        """
        ...


@dataclass()
class ModelMetadata:
    """Describes a trained model."""

    trained_at: datetime
    rasa_open_source_version: Text
    model_id: Text
    domain: Domain
    train_schema: GraphSchema
    predict_schema: GraphSchema
    training_type: TrainingType = TrainingType.BOTH

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serializable version of the `ModelMetadata`."""
        return {
            "domain": self.domain.as_dict(),
            "trained_at": self.trained_at.isoformat(),
            "model_id": self.model_id,
            "rasa_open_source_version": self.rasa_open_source_version,
            "train_schema": self.train_schema.as_dict(),
            "predict_schema": self.predict_schema.as_dict(),
            "training_type": self.training_type.value,
        }

    @classmethod
    def from_dict(cls, serialized: Dict[Text, Any]) -> ModelMetadata:
        """Loads `ModelMetadata` which has been serialized using `metadata.as_dict()`.

        Args:
            serialized: Serialized `ModelMetadata` (e.g. read from disk).

        Returns:
            Instantiated `ModelMetadata`.
        """
        from rasa.engine.graph import GraphSchema

        return ModelMetadata(
            trained_at=datetime.fromisoformat(serialized["trained_at"]),
            rasa_open_source_version=serialized["rasa_open_source_version"],
            model_id=serialized["model_id"],
            domain=Domain.from_dict(serialized["domain"]),
            train_schema=GraphSchema.from_dict(serialized["train_schema"]),
            predict_schema=GraphSchema.from_dict(serialized["predict_schema"]),
            training_type=TrainingType(serialized["training_type"]),
        )
