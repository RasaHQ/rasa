import dask
import pytest

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.config_to_graph import \
    (
    nlu_config_to_predict_graph_schema, old_config_to_predict_graph_schema,
    old_config_to_train_graph_schema,
)
from rasa.architecture_prototype.graph import _graph_component_for_config
from rasa.core.channels import UserMessage
from tests.architecture_prototype.conftest import clean_directory
from tests.architecture_prototype.graph_schema import predict_graph_schema

default_config = """
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 2
    constrain_similarities: true
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 2
    constrain_similarities: true
policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
    max_history: 5
    epochs: 10
    constrain_similarities: true
  - name: RulePolicy
"""

# TODO: put in config
"""
  - name: FallbackClassifier
    threshold: 0.3
    ambiguity_threshold: 0.1
"""

project = "examples/moodbot"


@pytest.mark.timeout(600)
def test_generate_train_graph():
    train_graph_schema, last_components_out = old_config_to_train_graph_schema(project=project, config=default_config)
    graph.fill_defaults(train_graph_schema)

    dask_graph = graph.convert_to_dask_graph(train_graph_schema)
    dask.visualize(dask_graph, filename="generated_default_config_graph.png")

    clean_directory()
    graph.run_as_dask_graph(
        train_graph_schema, last_components_out,
    )


@pytest.mark.timeout(600)
def test_generate_predict_graph():
    predict_graph_schema, targets = old_config_to_predict_graph_schema(config=default_config)
    graph.fill_defaults(predict_graph_schema)

    predict_graph = graph.convert_to_dask_graph(predict_graph_schema)

    predict_graph["load_user_message"] = _graph_component_for_config(
        "load_user_message",
        predict_graph_schema["load_user_message"],
        {"message":  UserMessage(text="hi")},
    )

    graph.visualise_dask_graph(predict_graph, 'generated_predict_graph')

    prediction = graph.run_dask_graph(predict_graph, targets)

