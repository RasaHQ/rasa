from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tarsafe import TarSafe
from typing import Generator, Optional, Text, Tuple, Union

import rasa.utils.common
import rasa.shared.utils.io
from rasa.engine.storage.storage import ModelMetadata, ModelStorage
from rasa.engine.graph import GraphModelConfiguration
from rasa.engine.storage.resource import Resource
from rasa.exceptions import UnsupportedModelVersionError
from rasa.shared.core.domain import Domain
import rasa.model

logger = logging.getLogger(__name__)

# Paths within model archive
MODEL_ARCHIVE_COMPONENTS_DIR = "components"
MODEL_ARCHIVE_METADATA_FILE = "metadata.json"


@contextmanager
def windows_safe_temporary_directory(
    suffix: Optional[Text] = None,
    prefix: Optional[Text] = None,
    dir: Optional[Text] = None,
) -> Generator[Text, None, None]:
    """Like `tempfile.TemporaryDirectory`, but works with Windows and long file names.

    On Windows by default there is a restriction on long path names.
    Using the prefix below allows to bypass this restriction in environments
    where it's not possible to override this behavior, mostly for internal
    policy reasons.

    Reference: https://stackoverflow.com/a/49102229
    """
    if sys.platform == "win32":
        directory = tempfile.mkdtemp(suffix, prefix, dir)
        directory = rasa.utils.common.decode_bytes(directory)

        try:
            yield directory
        finally:
            shutil.rmtree(f"\\\\?\\{directory}")
    else:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = rasa.utils.common.decode_bytes(temporary_directory)
            yield temporary_directory


class LocalModelStorage(ModelStorage):
    """Stores and provides output of `GraphComponents` on local disk."""

    def __init__(self, storage_path: Path) -> None:
        """Creates storage (see parent class for full docstring)."""
        self._storage_path = storage_path

    @classmethod
    def create(cls, storage_path: Path) -> ModelStorage:
        """Creates a new instance (see parent class for full docstring)."""
        return cls(storage_path)

    @classmethod
    def from_model_archive(
        cls, storage_path: Path, model_archive_path: Union[Text, Path]
    ) -> Tuple[LocalModelStorage, ModelMetadata]:
        """Initializes storage from archive (see parent class for full docstring)."""
        if next(storage_path.glob("*"), None):
            raise ValueError(
                f"The model storage with path '{storage_path}' is "
                f"not empty. You can only unpack model archives into an "
                f"empty model storage."
            )

        with windows_safe_temporary_directory() as temporary_directory:
            temporary_directory_path = Path(temporary_directory)

            cls._extract_archive_to_directory(
                model_archive_path, temporary_directory_path
            )
            logger.debug(f"Extracted model to '{temporary_directory_path}'.")
            cls._initialize_model_storage_from_model_archive(
                temporary_directory_path, storage_path
            )

            metadata = cls._load_metadata(temporary_directory_path)

            return (cls(storage_path), metadata)

    @classmethod
    def metadata_from_archive(
        cls, model_archive_path: Union[Text, Path]
    ) -> ModelMetadata:
        """Retrieves metadata from archive (see parent class for full docstring)."""
        with windows_safe_temporary_directory() as temporary_directory:
            temporary_directory_path = Path(temporary_directory)

            cls._extract_archive_to_directory(
                model_archive_path, temporary_directory_path
            )
            metadata = cls._load_metadata(temporary_directory_path)

            return metadata

    @staticmethod
    def _extract_archive_to_directory(
        model_archive_path: Union[Text, Path], temporary_directory: Path
    ) -> None:
        with TarSafe.open(model_archive_path, mode="r:gz") as tar:
            if sys.platform == "win32":
                # on Windows by default there is a restriction on long
                # path names; using the prefix below allows to bypass
                # this restriction in environments where it's not possible
                # to override this behavior, mostly for internal policy reasons
                # reference: https://stackoverflow.com/a/49102229
                tar.extractall(f"\\\\?\\{temporary_directory}")
            else:
                tar.extractall(temporary_directory)
        LocalModelStorage._assert_not_rasa2_archive(temporary_directory)

    @staticmethod
    def _assert_not_rasa2_archive(temporary_directory: Union[Text, Path]) -> None:
        fingerprint_file = Path(temporary_directory) / "fingerprint.json"
        if fingerprint_file.is_file():
            serialized_fingerprint = rasa.shared.utils.io.read_json_file(
                fingerprint_file
            )
            raise UnsupportedModelVersionError(
                model_version=serialized_fingerprint["version"]
            )

    @staticmethod
    def _initialize_model_storage_from_model_archive(
        temporary_directory: Path, storage_path: Path
    ) -> None:
        for path in (temporary_directory / MODEL_ARCHIVE_COMPONENTS_DIR).glob("*"):
            shutil.move(str(path), str(storage_path))

    @staticmethod
    def _load_metadata(directory: Path) -> ModelMetadata:
        serialized_metadata = rasa.shared.utils.io.read_json_file(
            directory / MODEL_ARCHIVE_METADATA_FILE
        )

        return ModelMetadata.from_dict(serialized_metadata)

    @contextmanager
    def write_to(self, resource: Resource) -> Generator[Path, None, None]:
        """Persists data for a resource (see parent class for full docstring)."""
        logger.debug(f"Resource '{resource.name}' was requested for writing.")
        directory = self._directory_for_resource(resource)

        if not directory.exists():
            directory.mkdir()

        yield directory

        logger.debug(f"Resource '{resource.name}' was persisted.")

    def _directory_for_resource(self, resource: Resource) -> Path:
        return self._storage_path / resource.name

    @contextmanager
    def read_from(self, resource: Resource) -> Generator[Path, None, None]:
        """Provides the data of a `Resource` (see parent class for full docstring)."""
        logger.debug(f"Resource '{resource.name}' was requested for reading.")
        directory = self._directory_for_resource(resource)

        if not directory.exists():
            raise ValueError(
                f"Resource '{resource.name}' does not exist. Please make "
                f"sure that the graph component providing the resource "
                f"is a parent node of the current graph node "
                f"(in case this happens during training) or that the "
                f"resource was actually persisted during training "
                f"(in case this happens during inference)."
            )

        yield directory

    def create_model_package(
        self,
        model_archive_path: Union[Text, Path],
        model_configuration: GraphModelConfiguration,
        domain: Domain,
    ) -> ModelMetadata:
        """Creates model package (see parent class for full docstring)."""
        logger.debug(f"Start to created model package for path '{model_archive_path}'.")

        with windows_safe_temporary_directory() as temp_dir:
            temporary_directory = Path(temp_dir)

            shutil.copytree(
                self._storage_path, temporary_directory / MODEL_ARCHIVE_COMPONENTS_DIR
            )

            model_metadata = self._create_model_metadata(domain, model_configuration)
            self._persist_metadata(model_metadata, temporary_directory)

            if isinstance(model_archive_path, str):
                model_archive_path = Path(model_archive_path)

            if not model_archive_path.parent.exists():
                model_archive_path.parent.mkdir(parents=True)

            with TarSafe.open(model_archive_path, "w:gz") as tar:
                tar.add(temporary_directory, arcname="")

        logger.debug(f"Model package created in path '{model_archive_path}'.")

        return model_metadata

    @staticmethod
    def _persist_metadata(metadata: ModelMetadata, temporary_directory: Path) -> None:
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            temporary_directory / MODEL_ARCHIVE_METADATA_FILE, metadata.as_dict()
        )

    @staticmethod
    def _create_model_metadata(
        domain: Domain, model_configuration: GraphModelConfiguration
    ) -> ModelMetadata:
        return ModelMetadata(
            trained_at=datetime.utcnow(),
            rasa_open_source_version=rasa.__version__,
            model_id=uuid.uuid4().hex,
            assistant_id=model_configuration.assistant_id,
            domain=domain,
            train_schema=model_configuration.train_schema,
            predict_schema=model_configuration.predict_schema,
            training_type=model_configuration.training_type,
            project_fingerprint=rasa.model.project_fingerprint(),
            spaces=model_configuration.spaces,
            language=model_configuration.language,
            core_target=model_configuration.core_target,
            nlu_target=model_configuration.nlu_target,
        )
