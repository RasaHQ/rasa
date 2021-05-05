from rasa.architecture_prototype.config_to_graph import (
    config_to_predict_graph_schema,
    config_to_train_graph_schema,
)
from tests.architecture_prototype.conftest import default_config, project
from tests.architecture_prototype.graph_schema import (
    predict_graph_schema,
    train_graph_schema,
)


def test_generate_train_graph():
    assert config_to_train_graph_schema(config=default_config) == train_graph_schema


def test_generate_predict_graph():
    assert config_to_predict_graph_schema(config=default_config) == predict_graph_schema
