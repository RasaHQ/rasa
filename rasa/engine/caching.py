from __future__ import annotations

import abc
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Text, Any, Optional, Tuple, List

from packaging import version
from sqlalchemy.engine import URL

from sqlalchemy.exc import OperationalError
from typing_extensions import Protocol, runtime_checkable

import rasa
import rasa.model
import rasa.utils.common
import rasa.shared.utils.common
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
import sqlalchemy as sa
import sqlalchemy.orm

from rasa.engine.storage.storage import ModelStorage
from rasa.shared.engine.caching import (
    get_local_cache_location,
    get_max_cache_size,
    get_cache_database_name,
)

logger = logging.getLogger(__name__)


class TrainingCache(abc.ABC):
    """Stores training results in a persistent cache.

    Used to minimize re-retraining when the data / config didn't change in between
    training runs.
    """

    @abc.abstractmethod
    def cache_output(
        self,
        fingerprint_key: Text,
        output: Any,
        output_fingerprint: Text,
        model_storage: ModelStorage,
    ) -> None:
        """Adds the output to the cache.

        If the output is of type `Cacheable` the output is persisted to disk in addition
        to its fingerprint.

        Args:
            fingerprint_key: The fingerprint key serves as key for the cache. Graph
                components can use their fingerprint key to lookup fingerprints of
                previous training runs.
            output: The output. The output is only cached to disk if it's of type
                `Cacheable`.
            output_fingerprint: The fingerprint of their output. This can be used
                to lookup potentially persisted outputs on disk.
            model_storage: Required for caching `Resource` instances. E.g. `Resource`s
                use that to copy data from the model storage to the cache.
        """

    ...

    @abc.abstractmethod
    def get_cached_output_fingerprint(self, fingerprint_key: Text) -> Optional[Text]:
        """Retrieves fingerprint of output based on fingerprint key.

        Args:
            fingerprint_key: The fingerprint serves as key for the lookup of output
                fingerprints.

        Returns:
            The fingerprint of a matching output or `None` in case no cache entry was
            found for the given fingerprint key.
        """
        ...

    @abc.abstractmethod
    def get_cached_result(
        self, output_fingerprint_key: Text, node_name: Text, model_storage: ModelStorage
    ) -> Optional[Cacheable]:
        """Returns a potentially cached output result.

        Args:
            output_fingerprint_key: The fingerprint key of the output serves as lookup
                key for a potentially cached version of this output.
            node_name: The name of the graph node which wants to use this cached result.
            model_storage: The current model storage (e.g. used when restoring
                `Resource` objects so that they can fill the model storage with data).

        Returns:
            `None` if no matching result was found or restored `Cacheable`.
        """
        ...


@runtime_checkable
class Cacheable(Protocol):
    """Protocol for cacheable graph component outputs.

    We only cache graph component outputs which are `Cacheable`. We only store the
    output fingerprint for everything else.
    """

    def to_cache(self, directory: Path, model_storage: ModelStorage) -> None:
        """Persists `Cacheable` to disk.

        Args:
            directory: The directory where the `Cacheable` can persist itself to.
            model_storage: The current model storage (e.g. used when caching `Resource`
                objects.
        """
        ...

    @classmethod
    def from_cache(
        cls,
        node_name: Text,
        directory: Path,
        model_storage: ModelStorage,
        output_fingerprint: Text,
    ) -> Cacheable:
        """Loads `Cacheable` from cache.

        Args:
            node_name: The name of the graph node which wants to use this cached result.
            directory: Directory containing the persisted `Cacheable`.
            model_storage: The current model storage (e.g. used when restoring
                `Resource` objects so that they can fill the model storage with data).
            output_fingerprint: The fingerprint of the cached result (e.g. used when
                restoring `Resource` objects as the fingerprint can not be easily
                calculated from the object itself).

        Returns:
            Instantiated `Cacheable`.
        """
        ...


class LocalTrainingCache(TrainingCache):
    """Caches training results on local disk (see parent class for full docstring)."""

    from sqlalchemy.orm import DeclarativeBase

    class Base(DeclarativeBase):
        pass

    class CacheEntry(Base):
        """Stores metadata about a single cache entry."""

        __tablename__ = "cache_entry"

        fingerprint_key = sa.Column(sa.String(), primary_key=True)
        output_fingerprint_key = sa.Column(sa.String(), nullable=False, index=True)
        last_used = sa.Column(sa.DateTime(timezone=True), nullable=False)
        rasa_version = sa.Column(sa.String(255), nullable=False)
        result_location = sa.Column(sa.String())
        result_type = sa.Column(sa.String())

    def __init__(self) -> None:
        """Creates cache.

        The `Cache` setting can be configured via environment variables.
        """
        self._cache_location = LocalTrainingCache._get_cache_location()

        self._max_cache_size = get_max_cache_size()

        self._cache_database_name = get_cache_database_name()

        if not self._cache_location.exists() and not self._is_disabled():
            logger.debug(
                f"Creating caching directory '{self._cache_location}' because "
                f"it doesn't exist yet."
            )
            self._cache_location.mkdir(parents=True)

        self._sessionmaker = self._create_database()

        self._drop_cache_entries_from_incompatible_versions()

    @staticmethod
    def _get_cache_location() -> Path:
        return get_local_cache_location()

    def _create_database(self) -> sqlalchemy.orm.sessionmaker:
        if self._is_disabled():
            # Use in-memory database as mock to avoid having to check `_is_disabled`
            # everywhere
            database = ""
        else:
            database = str(self._cache_location / self._cache_database_name)

        # Use `future=True` as we are using the 2.x query style
        engine = sa.create_engine(
            URL.create(drivername="sqlite", database=database), future=True
        )
        self.Base.metadata.create_all(engine)

        return sa.orm.sessionmaker(engine)

    def _drop_cache_entries_from_incompatible_versions(self) -> None:
        incompatible_entries = self._find_incompatible_cache_entries()

        for entry in incompatible_entries:
            self._delete_cached_result(entry)

        self._delete_incompatible_entries_from_cache(incompatible_entries)

        logger.debug(
            f"Deleted {len(incompatible_entries)} from disk as their version "
            f"is older than the minimum compatible version "
            f"('{MINIMUM_COMPATIBLE_VERSION}')."
        )

    def _find_incompatible_cache_entries(self) -> List[LocalTrainingCache.CacheEntry]:
        with self._sessionmaker() as session:
            query_for_cache_entries = sa.select(self.CacheEntry)
            all_entries: List[LocalTrainingCache.CacheEntry] = (
                session.execute(query_for_cache_entries).scalars().all()
            )

        return [
            entry
            for entry in all_entries
            if version.parse(MINIMUM_COMPATIBLE_VERSION)
            > version.parse(entry.rasa_version)
        ]

    def _delete_incompatible_entries_from_cache(
        self, incompatible_entries: List[LocalTrainingCache.CacheEntry]
    ) -> None:
        incompatible_fingerprints = [
            entry.fingerprint_key for entry in incompatible_entries
        ]
        with self._sessionmaker.begin() as session:
            delete_query = sa.delete(self.CacheEntry).where(
                self.CacheEntry.fingerprint_key.in_(incompatible_fingerprints)
            )
            session.execute(delete_query)

    @staticmethod
    def _delete_cached_result(entry: LocalTrainingCache.CacheEntry) -> None:
        if entry.result_location and Path(entry.result_location).is_dir():
            shutil.rmtree(entry.result_location)

    def cache_output(
        self,
        fingerprint_key: Text,
        output: Any,
        output_fingerprint: Text,
        model_storage: ModelStorage,
    ) -> None:
        """Adds the output to the cache (see parent class for full docstring)."""
        if self._is_disabled():
            return

        cache_dir, output_type = None, None
        if isinstance(output, Cacheable):
            cache_dir, output_type = self._cache_output_to_disk(output, model_storage)

        try:
            self._add_cache_entry(
                cache_dir, fingerprint_key, output_fingerprint, output_type
            )
        except OperationalError:
            if cache_dir:
                shutil.rmtree(cache_dir)

            raise

    def _add_cache_entry(
        self,
        cache_dir: Optional[Text],
        fingerprint_key: Text,
        output_fingerprint: Text,
        output_type: Text,
    ) -> None:
        with self._sessionmaker.begin() as session:
            cache_entry = self.CacheEntry(
                fingerprint_key=fingerprint_key,
                output_fingerprint_key=output_fingerprint,
                last_used=datetime.utcnow(),
                rasa_version=rasa.__version__,
                result_location=cache_dir,
                result_type=output_type,
            )
            session.merge(cache_entry)

    def _is_disabled(self) -> bool:
        return self._max_cache_size == 0.0

    def _cache_output_to_disk(
        self, output: Cacheable, model_storage: ModelStorage
    ) -> Tuple[Optional[Text], Optional[Text]]:
        tempdir_name = rasa.utils.common.get_temp_dir_name()

        # Use `TempDirectoryPath` instead of `tempfile.TemporaryDirectory` as this
        # leads to errors on Windows when the context manager tries to delete an
        # already deleted temporary directory (e.g. https://bugs.python.org/issue29982)
        with rasa.utils.common.TempDirectoryPath(tempdir_name) as temp_dir:
            tmp_path = Path(temp_dir)
            try:
                output.to_cache(tmp_path, model_storage)

                logger.debug(
                    f"Caching output of type '{type(output).__name__}' succeeded."
                )
            except Exception as e:
                logger.error(
                    f"Caching output of type '{type(output).__name__}' failed with the "
                    f"following error:\n{e}"
                )
                return None, None

            output_size = rasa.utils.common.directory_size_in_mb(tmp_path)
            if output_size > self._max_cache_size:
                logger.debug(
                    f"Caching result of type '{type(output).__name__}' was skipped "
                    f"because it exceeds the maximum cache size of "
                    f"{self._max_cache_size} MiB."
                )
                return None, None

            while (
                rasa.utils.common.directory_size_in_mb(
                    self._cache_location,
                    filenames_to_exclude=[self._cache_database_name],
                )
                + output_size
                > self._max_cache_size
            ):
                self._drop_least_recently_used_item()

            output_type = rasa.shared.utils.common.module_path_from_instance(output)
            cache_path = shutil.move(temp_dir, self._cache_location)

            return cache_path, output_type

    def _drop_least_recently_used_item(self) -> None:
        with self._sessionmaker.begin() as session:
            query_for_least_recently_used_entry = sa.select(self.CacheEntry).order_by(
                self.CacheEntry.last_used.asc()
            )
            oldest_cache_item = (
                session.execute(query_for_least_recently_used_entry).scalars().first()
            )

            if not oldest_cache_item:
                self._purge_cache_dir_content()
                return

            self._delete_cached_result(oldest_cache_item)
            delete_query = sa.delete(self.CacheEntry).where(
                self.CacheEntry.fingerprint_key == oldest_cache_item.fingerprint_key
            )
            session.execute(delete_query)

            logger.debug(
                f"Deleted item with fingerprint "
                f"'{oldest_cache_item.fingerprint_key}' to free space."
            )

    def _purge_cache_dir_content(self) -> None:
        for item in self._cache_location.glob("*"):
            if item.name == self._cache_database_name:
                continue

            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def get_cached_output_fingerprint(self, fingerprint_key: Text) -> Optional[Text]:
        """Returns cached output fingerprint (see parent class for full docstring)."""
        with self._sessionmaker.begin() as session:
            query = sa.select(self.CacheEntry).filter_by(
                fingerprint_key=fingerprint_key
            )
            match = session.execute(query).scalars().first()

            if match:
                # This result was used during a fingerprint run.
                match.last_used = datetime.utcnow()
                return match.output_fingerprint_key

            return None

    def get_cached_result(
        self, output_fingerprint_key: Text, node_name: Text, model_storage: ModelStorage
    ) -> Optional[Cacheable]:
        """Returns a potentially cached output (see parent class for full docstring)."""
        result_location, result_type = self._get_cached_result(output_fingerprint_key)

        if not result_location:
            logger.debug(f"No cached output found for '{output_fingerprint_key}'")
            return None

        path_to_cached = Path(result_location)
        if not path_to_cached.is_dir():
            logger.debug(
                f"Cached output for '{output_fingerprint_key}' can't be found on disk."
            )
            return None

        return self._load_from_cache(
            result_location,
            result_type,
            node_name,
            model_storage,
            output_fingerprint_key,
        )

    def _get_cached_result(
        self, output_fingerprint_key: Text
    ) -> Tuple[Optional[Path], Optional[Text]]:
        with self._sessionmaker.begin() as session:
            query = sa.select(
                self.CacheEntry.result_location, self.CacheEntry.result_type
            ).where(
                self.CacheEntry.output_fingerprint_key == output_fingerprint_key,
                self.CacheEntry.result_location != sa.null(),
            )

            match = session.execute(query).first()

            if match:
                return Path(match.result_location), match.result_type

            return None, None

    @staticmethod
    def _load_from_cache(
        path_to_cached: Path,
        result_type: Text,
        node_name: Text,
        model_storage: ModelStorage,
        output_fingerprint_key: Text,
    ) -> Optional[Cacheable]:
        try:
            module = rasa.shared.utils.common.class_from_module_path(result_type)

            if not isinstance(module, Cacheable):
                logger.warning(
                    "Failed to restore a non cacheable module from cache. "
                    "Please implement the 'Cacheable' interface for module "
                    f"'{result_type}'."
                )
                return None

            return module.from_cache(
                node_name, path_to_cached, model_storage, output_fingerprint_key
            )
        except Exception as e:
            logger.warning(
                f"Failed to restore cached output of type '{result_type}' from "
                f"cache. Error:\n{e}"
            )
            return None
