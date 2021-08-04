from pathlib import Path

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.graph_trainer import GraphTrainer
from tests.engine.graph_components_test_classes import PersistableTestComponent


def test_graph_trainer(
    default_model_storage: ModelStorage,
    temp_cache: TrainingCache,
    tmp_path: Path,
    domain_path: Path,
):
    graph_trainer = GraphTrainer(
        model_storage=default_model_storage,
        cache=temp_cache,
        graph_runner_class=DaskGraphRunner,
    )

    test_value_for_sub_directory = {"test": "test value sub dir"}
    test_value = {"test dir": "test value dir"}

    train_schema = GraphSchema(
        {
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={
                    "test_value": test_value,
                    "test_value_for_sub_directory": test_value_for_sub_directory,
                },
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
            "load": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="run_inference",
                constructor_name="load",
                config={},
                is_target=True,
                resource=Resource("train"),
            ),
        }
    )

    output_filename = tmp_path / "model.tar.gz"
    graph_runner = graph_trainer.train(
        train_schema=train_schema,
        predict_schema=predict_schema,
        domain_path=domain_path,
        output_filename=output_filename,
    )
    assert isinstance(graph_runner, DaskGraphRunner)
    assert output_filename.is_file()
    assert graph_runner.run() == {"load": test_value}

    # TODO: test unpacking from model package? (Or wait till graph_loader)


