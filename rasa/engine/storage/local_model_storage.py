from __future__ import annotations

import logging
import os
import shutil
import sys
import tarfile
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, List, Optional, Text, Tuple, Union

from tarsafe import TarSafe

import rasa.model
import rasa.shared.utils.io
import rasa.utils.common
from rasa.engine.graph import GraphModelConfiguration
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelMetadata, ModelStorage
from rasa.exceptions import UnsupportedModelVersionError
from rasa.shared.core.domain import Domain

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


def filter_normpath(member: tarfile.TarInfo, dest_path: str) -> tarfile.TarInfo:
    """Normalize tar member paths for safe extraction"""
    if member.name:
        member.name = os.path.normpath(member.name)
    return member


FilterFunction = Callable[[tarfile.TarInfo, str], Optional[tarfile.TarInfo]]


def create_combined_filter(existing_filter: Optional[FilterFunction]) -> FilterFunction:
    """Create a filter that combines existing filter with path normalization"""

    def combined_filter(
        member: tarfile.TarInfo, dest_path: str
    ) -> Optional[tarfile.TarInfo]:
        """Apply existing filter first, then path normalization"""
        if existing_filter is not None:
            filtered_member = existing_filter(member, dest_path)
            if filtered_member is None:
                return None  # Rejected by existing filter
            member = filtered_member  # Use the filtered result

        # Apply our path normalization
        return filter_normpath(member, dest_path)

    return combined_filter


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
                try:
                    # Use extraction filter to normalize paths for compatibility
                    # before trying the \\?\ prefix approach first
                    prev_filter = getattr(tar, "extraction_filter", None)
                    tar.extraction_filter = create_combined_filter(prev_filter)
                    tar.extractall(
                        f"\\\\?\\{temporary_directory}",
                        members=yield_safe_members(tar.getmembers()),
                    )
                except Exception:
                    # Fallback for Python versions with tarfile security fix
                    logger.warning(
                        "Failed to extract model archive with long path support. "
                        "Falling back to regular extraction."
                    )
                    tar.extractall(
                        temporary_directory,
                        members=yield_safe_members(tar.getmembers()),
                    )
            else:
                tar.extractall(
                    temporary_directory, members=yield_safe_members(tar.getmembers())
                )
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
            language=model_configuration.language,
            additional_languages=model_configuration.additional_languages,
            core_target=model_configuration.core_target,
            nlu_target=model_configuration.nlu_target,
        )


def yield_safe_members(
    members: List[tarfile.TarInfo],
) -> Generator[tarfile.TarInfo, None, None]:
    """
    Filter function for tar.extractall members parameter.
    Validates each member and yields only safe ones.

    Args:
        members: Iterator of TarInfo objects from tar.getmembers()

    Yields:
        TarInfo: Safe members to extract
    """
    for member in members:
        # Skip absolute paths
        if Path(member.name).is_absolute():
            continue

        # Skip paths with directory traversal sequences
        if ".." in member.name or "\\.." in member.name:
            continue

        # Skip special file types unless you need them
        if member.isdev() or member.issym():
            continue

        yield member
