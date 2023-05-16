import sys
import uuid
from datetime import datetime
from pathlib import Path
from tarsafe import TarSafe

import freezegun
import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory

import rasa.shared.utils.io
from rasa.engine.graph import SchemaNode, GraphSchema, GraphModelConfiguration
from rasa.engine.storage.local_model_storage import (
    LocalModelStorage,
    MODEL_ARCHIVE_METADATA_FILE,
)
from rasa.engine.storage.storage import ModelStorage, ModelMetadata
from rasa.engine.storage.resource import Resource
from rasa.exceptions import UnsupportedModelVersionError
from rasa.shared.core.domain import Domain
from rasa.shared.data import TrainingType
from tests.engine.graph_components_test_classes import PersistableTestComponent


def test_write_to_and_read(default_model_storage: ModelStorage):
    test_filename = "file.txt"
    test_file_content = "hi"

    test_sub_filename = "sub_file"
    test_sub_dir_name = "sub_directory"
    test_sub_file_content = "sub file"

    resource = Resource("some_node123")

    # Fill model storage for resource
    with default_model_storage.write_to(resource) as resource_directory:
        file = resource_directory / test_filename
        file.write_text(test_file_content)

        sub_directory = resource_directory / test_sub_dir_name
        sub_directory.mkdir()
        file_in_sub_directory = sub_directory / test_sub_filename
        file_in_sub_directory.write_text(test_sub_file_content)

    # Read written resource data from model storage to see whether all expected
    # content is there
    with default_model_storage.read_from(resource) as resource_directory:
        assert (resource_directory / test_filename).read_text() == test_file_content
        assert (
            resource_directory / test_sub_dir_name / test_sub_filename
        ).read_text() == test_sub_file_content


def test_read_from_not_existing_resource(default_model_storage: ModelStorage):
    with default_model_storage.write_to(Resource("resource1")) as temporary_directory:
        file = temporary_directory / "file.txt"
        file.write_text("test")

    with pytest.raises(ValueError):
        with default_model_storage.read_from(Resource("a different resource")) as _:
            pass


def test_read_from_rasa2_resource(tmp_path_factory: TempPathFactory):
    # we only search for the fingerprint file - nlu and core folder do not even need
    # to exist
    model_dir = tmp_path_factory.mktemp("model_dir")
    version = "2.8.5"
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        model_dir / "fingerprint.json",
        {"version": version, "irrelevant-other-key": "bla"},
    )

    model_zips = tmp_path_factory.mktemp("model_zips")
    resource_name = "model"
    with TarSafe.open(model_zips / resource_name, "w:gz") as tar:
        tar.add(model_dir, arcname="")

    storage_dir = tmp_path_factory.mktemp("storage_dir")
    storage = LocalModelStorage(storage_path=storage_dir)
    with pytest.raises(UnsupportedModelVersionError, match=f".*{version}.*"):
        storage.from_model_archive(
            storage_path=storage_dir, model_archive_path=model_zips / resource_name
        )
    with pytest.raises(UnsupportedModelVersionError, match=f".*{version}.*"):
        storage.metadata_from_archive(model_archive_path=model_zips / resource_name)


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="no need to run this test on other platform than Windows",
)
def test_read_long_resource_names_windows(
    tmp_path_factory: TempPathFactory,
    domain: Domain,
):
    model_dir = tmp_path_factory.mktemp("model_dir")
    version = "3.5.0"

    # full path length > 260 chars
    # but each component of the path needs to be below 255 chars
    # https://bugs.python.org/issue542314#msg10256
    long_file_path = model_dir / f"{'custom_file' * 22}.json"
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        Path(f"\\\\?\\{long_file_path}"),
        {"irrelevant-key": "bla"},
    )
    train_schema = GraphSchema(
        {
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={"some_config": 123455, "some more config": [{"nested": "hi"}]},
            ),
            "load": SchemaNode(
                needs={"resource": "train"},
                uses=PersistableTestComponent,
                fn="run_inference",
                constructor_name="load",
                config={},
                is_target=True,
            ),
        }
    )
    predict_schema = GraphSchema(
        {
            "run": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="run",
                constructor_name="load",
                config={"some_config": 123455, "some more config": [{"nested": "hi"}]},
            )
        }
    )
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        model_dir / MODEL_ARCHIVE_METADATA_FILE,
        ModelMetadata(
            trained_at=datetime.utcnow(),
            rasa_open_source_version=version,
            model_id="xxxxxxx",
            assistant_id="test_assistant",
            domain=domain,
            train_schema=train_schema,
            predict_schema=predict_schema,
            project_fingerprint="xxxxxxx",
            core_target="",
            nlu_target="",
            language="en",
            training_type=TrainingType.BOTH,
        ).as_dict(),
    )

    model_zips = tmp_path_factory.mktemp("model_zips")
    resource_name = "model"
    with TarSafe.open(model_zips / resource_name, "w:gz") as tar:
        tar.add(Path(f"\\\\?\\{model_dir}"), arcname="")

    storage_dir = tmp_path_factory.mktemp("storage_dir")
    storage = LocalModelStorage(storage_path=storage_dir)
    storage.metadata_from_archive(model_archive_path=model_zips / resource_name)


def test_create_model_package(tmp_path_factory: TempPathFactory, domain: Domain):
    train_model_storage = LocalModelStorage(
        tmp_path_factory.mktemp("train model storage")
    )

    train_schema = GraphSchema(
        {
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={"some_config": 123455, "some more config": [{"nested": "hi"}]},
            ),
            "load": SchemaNode(
                needs={"resource": "train"},
                uses=PersistableTestComponent,
                fn="run_inference",
                constructor_name="load",
                config={},
                is_target=True,
            ),
        }
    )

    predict_schema = GraphSchema(
        {
            "run": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="run",
                constructor_name="load",
                config={"some_config": 123455, "some more config": [{"nested": "hi"}]},
            )
        }
    )

    # Fill model Storage
    with train_model_storage.write_to(Resource("resource1")) as directory:
        file = directory / "file.txt"
        file.write_text("test")

    # Package model
    persisted_model_dir = tmp_path_factory.mktemp("persisted models")
    archive_path = persisted_model_dir / "my-model.tar.gz"

    trained_at = datetime.utcnow()
    with freezegun.freeze_time(trained_at):
        train_model_storage.create_model_package(
            archive_path,
            GraphModelConfiguration(
                train_schema,
                predict_schema,
                TrainingType.BOTH,
                "test_assistant",
                None,
                None,
                "nlu",
            ),
            domain,
        )

    # Unpack and inspect packaged model
    load_model_storage_dir = tmp_path_factory.mktemp("load model storage")

    just_packaged_metadata = LocalModelStorage.metadata_from_archive(archive_path)

    (load_model_storage, packaged_metadata) = LocalModelStorage.from_model_archive(
        load_model_storage_dir, archive_path
    )

    assert just_packaged_metadata.trained_at == packaged_metadata.trained_at

    assert packaged_metadata.train_schema == train_schema
    assert packaged_metadata.predict_schema == predict_schema
    assert packaged_metadata.domain.as_dict() == domain.as_dict()

    assert packaged_metadata.rasa_open_source_version == rasa.__version__
    assert packaged_metadata.trained_at == trained_at
    assert packaged_metadata.model_id
    assert packaged_metadata.assistant_id == "test_assistant"
    assert packaged_metadata.project_fingerprint

    persisted_resources = load_model_storage_dir.glob("*")
    assert list(persisted_resources) == [Path(load_model_storage_dir, "resource1")]


def test_read_unsupported_model(
    monkeypatch: MonkeyPatch, tmp_path_factory: TempPathFactory, domain: Domain
):
    train_model_storage = LocalModelStorage(
        tmp_path_factory.mktemp("train model storage")
    )
    graph_schema = GraphSchema(nodes={})

    persisted_model_dir = tmp_path_factory.mktemp("persisted models")
    archive_path = persisted_model_dir / "my-model.tar.gz"

    # Create outdated model meta data
    trained_at = datetime.utcnow()
    model_configuration = GraphModelConfiguration(
        graph_schema,
        graph_schema,
        TrainingType.BOTH,
        "test_assistant",
        None,
        None,
        "nlu",
    )
    outdated_model_meta_data = ModelMetadata(
        trained_at=trained_at,
        rasa_open_source_version=rasa.__version__,  # overwrite later to avoid error
        model_id=uuid.uuid4().hex,
        assistant_id=model_configuration.assistant_id,
        domain=domain,
        train_schema=model_configuration.train_schema,
        predict_schema=model_configuration.predict_schema,
        training_type=model_configuration.training_type,
        project_fingerprint=rasa.model.project_fingerprint(),
        language=model_configuration.language,
        core_target=model_configuration.core_target,
        nlu_target=model_configuration.nlu_target,
    )
    old_version = "0.0.1"
    outdated_model_meta_data.rasa_open_source_version = old_version

    # Package model - and inject the outdated model meta data
    monkeypatch.setattr(
        LocalModelStorage,
        "_create_model_metadata",
        lambda *args, **kwargs: outdated_model_meta_data,
    )
    train_model_storage.create_model_package(
        model_archive_path=archive_path,
        model_configuration=model_configuration,
        domain=domain,
    )

    # Unpack and inspect packaged model
    load_model_storage_dir = tmp_path_factory.mktemp("load model storage")

    expected_message = (
        f"The model version is trained using Rasa Open Source "
        f"{old_version} and is not compatible with your current "
        f"installation .*"
    )
    with pytest.raises(UnsupportedModelVersionError, match=expected_message):
        LocalModelStorage.metadata_from_archive(archive_path)

    with pytest.raises(UnsupportedModelVersionError, match=expected_message):
        LocalModelStorage.from_model_archive(load_model_storage_dir, archive_path)


def test_create_package_with_non_existing_parent(tmp_path: Path):
    storage = LocalModelStorage.create(tmp_path)
    model_file = tmp_path / "new" / "sub" / "dir" / "file.tar.gz"

    storage.create_model_package(
        model_file,
        GraphModelConfiguration(
            GraphSchema({}),
            GraphSchema({}),
            TrainingType.BOTH,
            "test_assistant",
            None,
            None,
            "nlu",
        ),
        Domain.empty(),
    )

    assert model_file.is_file()


def test_create_model_package_with_non_empty_model_storage(tmp_path: Path):
    # Put something in the future model storage directory
    (tmp_path / "somefile.json").touch()

    with pytest.raises(ValueError):
        # Unpacking into an already filled `ModelStorage` raises an exception.
        _ = LocalModelStorage.from_model_archive(tmp_path, Path("does not matter"))


def test_create_model_package_with_non_existing_dir(
    tmp_path: Path, default_model_storage: ModelStorage
):
    path = tmp_path / "some_dir" / "another" / "model.tar.gz"
    default_model_storage.create_model_package(
        path,
        GraphModelConfiguration(
            GraphSchema({}),
            GraphSchema({}),
            TrainingType.BOTH,
            "test_assistant",
            None,
            None,
            "nlu",
        ),
        Domain.empty(),
    )

    assert path.exists()
