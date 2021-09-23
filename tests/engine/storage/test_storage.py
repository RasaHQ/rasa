from datetime import datetime
from pathlib import Path

import rasa.shared.utils.io
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.engine.storage.storage import ModelMetadata
from rasa.shared.core.domain import Domain
from tests.engine.graph_components_test_classes import PersistableTestComponent


def test_metadata_serialization(domain: Domain, tmp_path: Path):
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
            ),
        }
    )

    trained_at = datetime.utcnow()
    rasa_version = "3.0.0"
    model_id = "some unique model id"
    metadata = ModelMetadata(
        trained_at, rasa_version, model_id, domain, train_schema, predict_schema
    )

    serialized = metadata.as_dict()

    # Dump and Load to make sure it's serializable
    dump_path = tmp_path / "metadata.json"
    rasa.shared.utils.io.dump_obj_as_json_to_file(dump_path, serialized)
    loaded_serialized = rasa.shared.utils.io.read_json_file(dump_path)

    loaded_metadata = ModelMetadata.from_dict(loaded_serialized)

    assert loaded_metadata.domain.as_dict() == domain.as_dict()
    assert loaded_metadata.model_id == model_id
    assert loaded_metadata.rasa_open_source_version == rasa_version
    assert loaded_metadata.trained_at == trained_at
    assert loaded_metadata.train_schema == train_schema
    assert loaded_metadata.predict_schema == predict_schema
