import shutil
from pathlib import Path

import dask

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.config_to_graph import nlu_config_to_train_graph

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
    dask.visualize(dask_graph, filename="graph.png")

    # clean up before testing persistence
    cache_dir = Path("model")
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir()

    graph.run_as_dask_graph(
        nlu_train_graph, [last_component_out],
    )
