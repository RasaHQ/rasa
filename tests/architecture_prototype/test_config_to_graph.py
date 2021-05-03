import dask
import pytest

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.config_to_graph import old_config_to_graph_schema
from tests.architecture_prototype.conftest import clean_directory

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

project = "examples/moodbot"


@pytest.mark.timeout(600)
def test_generate_train_graph():
    train_graph_schema, last_components_out = old_config_to_graph_schema(project=project, config=default_config)
    graph.fill_defaults(train_graph_schema)

    dask_graph = graph.convert_to_dask_graph(train_graph_schema)
    dask.visualize(dask_graph, filename="generated_default_config_graph.png")

    clean_directory()
    graph.run_as_dask_graph(
        train_graph_schema, last_components_out,
    )

