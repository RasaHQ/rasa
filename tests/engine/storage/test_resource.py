from pathlib import Path

import pytest
from _pytest.tmpdir import TempPathFactory

import rasa.utils.common
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage


def test_resource_caching(tmp_path_factory: TempPathFactory):
    model_storage = LocalModelStorage(tmp_path_factory.mktemp("initial_model_storage"))

    resource = Resource("my resource")

    # Fill model storage
    test_filename = "file.txt"
    test_content = "test_resource_caching"
    with model_storage.write_to(resource) as temporary_directory:
        file = temporary_directory / test_filename
        file.write_text(test_content)

    cache_dir = tmp_path_factory.mktemp("cache_dir")

    # Cache resource
    resource.to_cache(cache_dir, model_storage)

    # Reload resource from cache and inspect
    new_model_storage = LocalModelStorage(tmp_path_factory.mktemp("new_model_storage"))
    reinstantiated_resource = Resource.from_cache(
        resource.name, cache_dir, new_model_storage
    )

    assert reinstantiated_resource == resource

    # Read written resource data from model storage to see whether all expected
    # contents are there
    with new_model_storage.read_from(resource) as temporary_directory:
        assert (temporary_directory / test_filename).read_text() == test_content


def test_resource_caching_if_already_restored(tmp_path_factory: TempPathFactory):
    initial_storage_dir = tmp_path_factory.mktemp("initial_model_storage")
    model_storage = LocalModelStorage(initial_storage_dir)

    resource = Resource("my resource")

    # Fill model storage
    test_filename = "file.txt"
    test_content = "test_resource_caching"
    with model_storage.write_to(resource) as temporary_directory:
        file = temporary_directory / test_filename
        file.write_text(test_content)

    cache_dir = tmp_path_factory.mktemp("cache_dir")

    # Cache resource
    resource.to_cache(cache_dir, model_storage)

    new_storage_dir = tmp_path_factory.mktemp("new dir")
    rasa.utils.common.copy_directory(initial_storage_dir, new_storage_dir)

    reinstantiated_resource = Resource.from_cache(
        resource.name, cache_dir, LocalModelStorage(new_storage_dir)
    )

    assert reinstantiated_resource == resource


def test_resource_caching_if_already_restored_with_different_state(
    tmp_path_factory: TempPathFactory,
):
    initial_storage_dir = tmp_path_factory.mktemp("initial_model_storage")
    model_storage = LocalModelStorage(initial_storage_dir)

    resource = Resource("my resource")

    # Fill model storage
    test_filename = "file.txt"
    test_content = "test_resource_caching"
    with model_storage.write_to(resource) as temporary_directory:
        file = temporary_directory / test_filename
        file.write_text(test_content)

    cache_dir = tmp_path_factory.mktemp("cache_dir")

    # Cache resource
    resource.to_cache(cache_dir, model_storage)

    # Pretend there is an additional file which is not in the restored storage.
    # This makes the directories and causes the `from_cache` part to fail
    # different
    (temporary_directory / "another_file").touch()

    new_storage_dir = tmp_path_factory.mktemp("new dir")
    rasa.utils.common.copy_directory(initial_storage_dir, new_storage_dir)

    with pytest.raises(ValueError):
        Resource.from_cache(
            resource.name, cache_dir, LocalModelStorage(new_storage_dir)
        )


def test_resource_fingerprinting():
    resource1 = Resource("resource 1")
    resource2 = Resource("resource 2")

    fingerprint1 = resource1.fingerprint()
    fingerprint2 = resource2.fingerprint()

    assert fingerprint1
    assert fingerprint2

    assert fingerprint1 != fingerprint2


def test_caching_empty_resource(default_model_storage: ModelStorage, tmp_path: Path):
    resource = Resource("my resource")

    # does not raise
    resource.to_cache(tmp_path, default_model_storage)
