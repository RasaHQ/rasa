from __future__ import annotations
import abc
import logging
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union, Text, Generator, Dict, Any, Optional
from packaging import version

from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.exceptions import UnsupportedModelVersionError
from rasa.engine.storage.resource import Resource
from rasa.shared.core.domain import Domain
from rasa.shared.data import TrainingType

if typing.TYPE_CHECKING:
    from rasa.engine.graph import GraphSchema, GraphModelConfiguration

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

        Raises:
            `UnsupportedModelError` if the loaded meta data indicates that the model
            has been created with an outdated Rasa version.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def metadata_from_archive(
        cls, model_archive_path: Union[Text, Path]
    ) -> ModelMetadata:
        """Retrieves metadata from archive.

        Args:
            model_archive_path: The path to the model archive.

        Returns:
            Metadata about the model.

        Raises:
            `UnsupportedModelError` if the loaded meta data indicates that the model
            has been created with an outdated Rasa version.
        """
        ...

    @contextmanager
    @abc.abstractmethod
    def write_to(self, resource: Resource) -> Generator[Path, None, None]:
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
    def read_from(self, resource: Resource) -> Generator[Path, None, None]:
        """Provides the data of a persisted `Resource`.

        Args:
            resource: The `Resource` whose persisted should be accessed.

        Returns:
            A directory containing the data of the persisted `Resource`.

        Raises:
            ValueError: In case no persisted data for the given `Resource` exists.
        """
        ...

    @abc.abstractmethod
    def create_model_package(
        self,
        model_archive_path: Union[Text, Path],
        model_configuration: GraphModelConfiguration,
        domain: Domain,
    ) -> ModelMetadata:
        """Creates a model archive containing all data to load and run the model.

        Args:
            model_archive_path: The path to the archive which should be created.
            model_configuration: The model configuration (schemas, language, etc.)
            domain: The `Domain` which was used to train the model.

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
    assistant_id: Optional[Text]
    domain: Domain
    train_schema: GraphSchema
    predict_schema: GraphSchema
    project_fingerprint: Text
    core_target: Optional[Text]
    nlu_target: Text
    language: Optional[Text]
    spaces: Optional[List[Dict[Text, Any]]] = None
    training_type: TrainingType = TrainingType.BOTH

    def __post_init__(self) -> None:
        """Raises an exception when the metadata indicates an unsupported version.

        Raises:
            `UnsupportedModelException` if the `rasa_open_source_version` is lower
            than the minimum compatible version
        """
        minimum_version = version.parse(MINIMUM_COMPATIBLE_VERSION)
        model_version = version.parse(self.rasa_open_source_version)
        if model_version < minimum_version:
            raise UnsupportedModelVersionError(model_version=model_version)

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serializable version of the `ModelMetadata`."""
        return {
            "domain": self.domain.as_dict(),
            "trained_at": self.trained_at.isoformat(),
            "model_id": self.model_id,
            "assistant_id": self.assistant_id,
            "rasa_open_source_version": self.rasa_open_source_version,
            "train_schema": self.train_schema.as_dict(),
            "predict_schema": self.predict_schema.as_dict(),
            "training_type": self.training_type.value,
            "project_fingerprint": self.project_fingerprint,
            "core_target": self.core_target,
            "nlu_target": self.nlu_target,
            "language": self.language,
            "spaces": self.spaces,
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
            assistant_id=serialized.get("assistant_id"),
            domain=Domain.from_dict(serialized["domain"]),
            train_schema=GraphSchema.from_dict(serialized["train_schema"]),
            predict_schema=GraphSchema.from_dict(serialized["predict_schema"]),
            training_type=TrainingType(serialized["training_type"]),
            project_fingerprint=serialized["project_fingerprint"],
            core_target=serialized["core_target"],
            nlu_target=serialized["nlu_target"],
            language=serialized["language"],
            # optional, since introduced later
            spaces=serialized.get("spaces"),
        )
