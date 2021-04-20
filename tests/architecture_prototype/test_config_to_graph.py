import dask

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.config_to_graph import nlu_config_to_train_graph
from tests.architecture_prototype.test_graph import clean_directory

nlu_config = """
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
"""


def test_train_nlu():
    nlu_train_graph, last_component_out = nlu_config_to_train_graph(nlu_config)
    dask_graph = graph.convert_to_dask_graph(nlu_train_graph)
    dask.visualize(dask_graph, filename="auto_generated_nlu_graph.png")

    clean_directory()

    graph.run_as_dask_graph(
        nlu_train_graph, [last_component_out],
    )
