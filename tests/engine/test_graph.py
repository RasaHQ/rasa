from __future__ import annotations

from pathlib import Path
from typing import Dict, Text

import pytest

from rasa.engine.storage.resource import Resource
import rasa.shared.utils.io
from rasa.engine.exceptions import GraphSchemaException
from rasa.engine.graph import SchemaNode, GraphSchema
from tests.engine.graph_components_test_classes import PersistableTestComponent


def test_serialize_graph_schema(tmp_path: Path):
    graph_schema = GraphSchema(
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
                resource=Resource("test resource"),
            ),
        }
    )

    serialized = graph_schema.as_dict()

    # Dump it to make sure it's actually serializable
    file_path = tmp_path / "my_graph.yml"
    rasa.shared.utils.io.write_yaml(serialized, file_path)

    serialized_graph_schema_from_file = rasa.shared.utils.io.read_yaml_file(file_path)
    graph_schema_from_file = GraphSchema.from_dict(serialized_graph_schema_from_file)

    assert graph_schema_from_file == graph_schema


def test_invalid_module_error_when_deserializing_schemas(tmp_path: Path):
    graph_schema = GraphSchema(
        {
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={"some_config": 123455, "some more config": [{"nested": "hi"}]},
            )
        }
    )

    serialized = graph_schema.as_dict()

    # Pretend module is for some reason invalid
    serialized["nodes"]["train"]["uses"] = "invalid.class"

    # Dump it to make sure it's actually serializable
    file_path = tmp_path / "my_graph.yml"
    rasa.shared.utils.io.write_yaml(serialized, file_path)

    serialized_graph_schema_from_file = rasa.shared.utils.io.read_yaml_file(file_path)

    with pytest.raises(GraphSchemaException):
        _ = GraphSchema.from_dict(serialized_graph_schema_from_file)


def test_minimal_graph_schema():
    def test_schema_node(needs: Dict[Text, Text], target: bool = False) -> SchemaNode:
        return SchemaNode(
            needs=needs,
            uses=None,
            fn="",
            constructor_name="",
            config={},
            is_target=target,
        )

    test_schema = GraphSchema(
        {
            "1": test_schema_node({"i": "3"}, True),
            "2": test_schema_node({"i": "3"}),
            "3": test_schema_node({"i": "4"}),
            "4": test_schema_node({}),
            "5": test_schema_node({"i": "6"}),
            "6": test_schema_node({}),
            "7": test_schema_node({}),
            "8": test_schema_node({"i": "9"}, True),
            "9": test_schema_node({"i": "__input__"}),
        }
    )
    assert test_schema.minimal_graph_schema() == GraphSchema(
        {
            "1": test_schema_node({"i": "3"}, True),
            "3": test_schema_node({"i": "4"}),
            "4": test_schema_node({}),
            "8": test_schema_node({"i": "9"}, True),
            "9": test_schema_node({"i": "__input__"}),
        }
    )

    assert test_schema.minimal_graph_schema(targets=["8"]) == GraphSchema(
        {
            "8": test_schema_node({"i": "9"}, True),
            "9": test_schema_node({"i": "__input__"}),
        }
    )
