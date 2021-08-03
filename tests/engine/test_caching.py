import dataclasses
import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, Text

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from sqlalchemy.exc import OperationalError

import rasa.shared.utils.io
from rasa.engine.caching import (
    LocalTrainingCache,
    CACHE_LOCATION_ENV,
    DEFAULT_CACHE_NAME,
    CACHE_SIZE_ENV,
    CACHE_DB_NAME_ENV,
    TrainingCache,
)
import tests.conftest
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelStorage


@pytest.fixture()
def temp_cache(tmp_path: Path, monkeypatch: MonkeyPatch) -> LocalTrainingCache:
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))
    return LocalTrainingCache()


@dataclasses.dataclass
class TestCacheableOutput:

    value: Dict
    size_in_mb: int = 0

    def to_cache(self, directory: Path, model_storage: ModelStorage) -> None:
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            directory / "cached.json", self.value
        )

        # Can be used to create cache results of desired size
        if self.size_in_mb:
            tests.conftest.create_test_file_with_size(directory, self.size_in_mb)

    @classmethod
    def from_cache(
        cls, node_name: Text, directory: Path, model_storage: ModelStorage
    ) -> "TestCacheableOutput":
        value = rasa.shared.utils.io.read_json_file(directory / "cached.json")

        return cls(value)


def test_cache_output(tmp_path: Path, temp_cache: TrainingCache):
    model_storage = LocalModelStorage(tmp_path)

    fingerprint_key = uuid.uuid4().hex
    output = TestCacheableOutput({"something to cache": "dasdaasda"})
    output_fingerprint = uuid.uuid4().hex

    temp_cache.cache_output(fingerprint_key, output, output_fingerprint, model_storage)

    assert (
        temp_cache.get_cached_output_fingerprint(fingerprint_key) == output_fingerprint
    )

    cached = temp_cache.get_cached_result(output_fingerprint)

    assert cached.cached_type == TestCacheableOutput
    assert (
        cached.cached_type.from_cache("xy", cached.cache_directory, model_storage)
        == output
    )


def test_get_cached_result_with_miss(
    tmp_path: Path, temp_cache: TrainingCache, default_model_storage: ModelStorage
):
    # Cache something
    temp_cache.cache_output(
        uuid.uuid4().hex,
        TestCacheableOutput({"something to cache": "dasdaasda"}),
        uuid.uuid4().hex,
        default_model_storage,
    )

    assert temp_cache.get_cached_result(uuid.uuid4().hex) is None
    assert temp_cache.get_cached_output_fingerprint(uuid.uuid4().hex) is None


def test_get_cached_result_when_result_no_longer_available(
    tmp_path: Path, monkeypatch: MonkeyPatch, default_model_storage: ModelStorage
):
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    cache = LocalTrainingCache()

    output = TestCacheableOutput({"something to cache": "dasdaasda"})
    output_fingerprint = uuid.uuid4().hex

    cache.cache_output(
        uuid.uuid4().hex, output, output_fingerprint, default_model_storage
    )

    # Pretend something deleted the cache in between
    for path in tmp_path.glob("*"):
        if path.is_dir():
            shutil.rmtree(path)

    assert cache.get_cached_result(output_fingerprint) is None


def test_cache_creates_location_if_missing(tmp_path: Path, monkeypatch: MonkeyPatch):
    cache_location = tmp_path / "directory does not exist yet"

    monkeypatch.setenv(CACHE_LOCATION_ENV, str(cache_location))

    _ = LocalTrainingCache()

    assert cache_location.is_dir()


def test_caching_something_which_is_not_cacheable(
    tmp_path: Path, temp_cache: TrainingCache, default_model_storage: ModelStorage
):
    # Cache something
    fingerprint_key = uuid.uuid4().hex
    output_fingerprint_key = uuid.uuid4().hex
    temp_cache.cache_output(
        fingerprint_key, None, output_fingerprint_key, default_model_storage
    )

    # Output fingerprint was saved
    assert (
        temp_cache.get_cached_output_fingerprint(fingerprint_key)
        == output_fingerprint_key
    )

    # But it's not stored to disk
    assert temp_cache.get_cached_result(output_fingerprint_key) is None


def test_caching_cacheable_fails(
    tmp_path: Path,
    caplog: LogCaptureFixture,
    temp_cache: TrainingCache,
    default_model_storage: ModelStorage,
):
    fingerprint_key = uuid.uuid4().hex

    # `tmp_path` is not a dict and will hence fail to be cached
    # noinspection PyTypeChecker
    output = TestCacheableOutput(tmp_path)
    output_fingerprint = uuid.uuid4().hex

    with caplog.at_level(logging.ERROR):
        temp_cache.cache_output(
            fingerprint_key, output, output_fingerprint, default_model_storage
        )

    assert len(caplog.records) == 1

    assert (
        temp_cache.get_cached_output_fingerprint(fingerprint_key) == output_fingerprint
    )

    assert temp_cache.get_cached_result(output_fingerprint) is None


def test_removing_no_longer_compatible_cache_entries(
    tmp_path: Path, monkeypatch: MonkeyPatch, default_model_storage: ModelStorage
):
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    cache = LocalTrainingCache()

    # Cache an entry including serialized output which will be incompatible later
    fingerprint_key1 = uuid.uuid4().hex
    output1 = TestCacheableOutput({"something to cache": "dasdaasda"})
    output_fingerprint1 = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key1, output1, output_fingerprint1, default_model_storage
    )

    # Cache an entry without serialized output which will be incompatible later
    fingerprint_key2 = uuid.uuid4().hex
    output_fingerprint2 = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key2, None, output_fingerprint2, default_model_storage
    )

    # Cache a second entry with a newer Rasa version
    monkeypatch.setattr(rasa, "__version__", "99999.9.5")
    fingerprint_key3 = uuid.uuid4().hex
    output3 = TestCacheableOutput({"something to cache2": "dasdaasda"})
    output_fingerprint3 = uuid.uuid4().hex

    cache.cache_output(
        fingerprint_key3, output3, output_fingerprint3, default_model_storage
    )

    # Pretend we updated Rasa Open Source to a no longer compatible version
    monkeypatch.setattr(rasa.engine.caching, "MINIMUM_COMPATIBLE_VERSION", "99999.8.10")

    cache_run_by_future_rasa = LocalTrainingCache()

    # Cached fingerprints can no longer be retrieved
    assert (
        cache_run_by_future_rasa.get_cached_output_fingerprint(fingerprint_key1) is None
    )
    assert (
        cache_run_by_future_rasa.get_cached_output_fingerprint(fingerprint_key2) is None
    )

    assert cache_run_by_future_rasa.get_cached_result(output_fingerprint1) is None
    assert cache_run_by_future_rasa.get_cached_result(output_fingerprint2) is None

    # Entry 3 wasn't deleted from cache as it's still compatible
    assert (
        cache_run_by_future_rasa.get_cached_output_fingerprint(fingerprint_key3)
        == output_fingerprint3
    )
    cached_output = cache_run_by_future_rasa.get_cached_result(output_fingerprint3)
    assert (
        cached_output.cached_type.from_cache(
            "some node", cached_output.cache_directory, default_model_storage
        )
        == output3
    )

    # Cached output of no longer compatible stuff was deleted from disk
    assert set(tmp_path.glob("*")) == {
        tmp_path / DEFAULT_CACHE_NAME,
        cached_output.cache_directory,
    }


def test_skip_caching_if_cache_size_is_zero(
    tmp_path: Path, monkeypatch: MonkeyPatch, default_model_storage: ModelStorage
):
    cache_location = tmp_path / "cache"
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(cache_location))

    # Disable cache
    monkeypatch.setenv(CACHE_SIZE_ENV, "0")

    cache = LocalTrainingCache()

    # Cache something
    fingerprint_key1 = uuid.uuid4().hex
    output1 = TestCacheableOutput({"something to cache": "dasdaasda"})
    output_fingerprint1 = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key1, output1, output_fingerprint1, default_model_storage
    )

    # not even the database and no subdirectory was created ⛔️
    assert list(tmp_path.glob("*")) == []

    assert cache.get_cached_output_fingerprint(fingerprint_key1) is None

    assert cache.get_cached_result(output_fingerprint1) is None


def test_skip_caching_if_result_exceeds_max_size(
    tmp_path: Path, monkeypatch: MonkeyPatch, default_model_storage: ModelStorage
):
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    # Pretend we have a cache of size "1"
    monkeypatch.setenv(CACHE_SIZE_ENV, "1")

    cache = LocalTrainingCache()

    # Cache something
    fingerprint_key1 = uuid.uuid4().hex
    output1 = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint1 = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key1, output1, output_fingerprint1, default_model_storage
    )

    assert cache.get_cached_output_fingerprint(fingerprint_key1) == output_fingerprint1

    assert cache.get_cached_result(output_fingerprint1) is None


def test_delete_using_lru_if_cache_exceeds_size(
    tmp_path: Path, monkeypatch: MonkeyPatch, default_model_storage: ModelStorage
):
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    # Pretend we have a cache of certain size
    monkeypatch.setenv(CACHE_SIZE_ENV, "5")

    cache = LocalTrainingCache()

    # Cache an item
    fingerprint_key1 = uuid.uuid4().hex
    output1 = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint1 = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key1, output1, output_fingerprint1, default_model_storage
    )

    # Cache an non cacheable item to spice it up 🔥
    fingerprint_key2 = uuid.uuid4().hex
    output2 = TestCacheableOutput(None)
    output_fingerprint2 = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key2, output2, output_fingerprint2, default_model_storage
    )

    # Cache another item
    fingerprint_key3 = uuid.uuid4().hex
    output3 = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint3 = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key3, output3, output_fingerprint3, default_model_storage
    )

    # Assert both are there
    for output_fingerprint in [output_fingerprint1, output_fingerprint2]:
        assert cache.get_cached_result(output_fingerprint).cache_directory

    # Checkout the first item as this updates `last_used` and hence affects LRU
    cache.get_cached_output_fingerprint(fingerprint_key1)

    # Now store something which requires a deletion
    fingerprint_key4 = uuid.uuid4().hex
    output4 = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint4 = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key4, output4, output_fingerprint4, default_model_storage
    )

    # Assert cached result 1 and 3 are there
    for output_fingerprint in [output_fingerprint1, output_fingerprint4]:
        assert cache.get_cached_result(output_fingerprint).cache_directory

    # Cached result 2 and 3 were deleted
    assert cache.get_cached_output_fingerprint(fingerprint_key2) is None
    assert cache.get_cached_result(output_fingerprint3) is None


def test_cache_exceeds_size_but_not_in_database(
    tmp_path: Path, monkeypatch: MonkeyPatch, default_model_storage: ModelStorage
):
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    max_cache_size = 5
    # Pretend we have a cache of size `max_cached_size`
    monkeypatch.setenv(CACHE_SIZE_ENV, str(max_cache_size))

    # Fill cache with something which is not in the cache metadata
    sub_dir = tmp_path / "some dir"
    sub_dir.mkdir()

    # one subdirectory which needs deletion
    tests.conftest.create_test_file_with_size(sub_dir, max_cache_size)
    # one file which needs deletion
    tests.conftest.create_test_file_with_size(tmp_path, max_cache_size)

    cache = LocalTrainingCache()

    # Cache an item
    fingerprint_key = uuid.uuid4().hex
    output = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint = uuid.uuid4().hex
    cache.cache_output(
        fingerprint_key, output, output_fingerprint, default_model_storage
    )

    assert cache.get_cached_output_fingerprint(fingerprint_key) == output_fingerprint
    assert cache.get_cached_result(output_fingerprint).cache_directory


def test_clean_up_of_cached_result_if_database_fails(
    tmp_path: Path, monkeypatch: MonkeyPatch, default_model_storage: ModelStorage
):
    database_name = "test.db"
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))
    monkeypatch.setenv(CACHE_DB_NAME_ENV, database_name)

    cache = LocalTrainingCache()

    # Deleting the database will cause an error when caching the result
    (tmp_path / database_name).unlink()

    # Cache an item
    fingerprint_key = uuid.uuid4().hex
    output = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint = uuid.uuid4().hex

    with pytest.raises(OperationalError):
        cache.cache_output(
            fingerprint_key, output, output_fingerprint, default_model_storage
        )

    assert list(tmp_path.glob("*")) == [tmp_path / database_name]
