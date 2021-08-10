from pathlib import Path
from typing import Tuple, Type

from rasa.engine.graph import ExecutionContext
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelMetadata, ModelStorage


def load_predict_graph_runner(
    storage_path: Path,
    model_archive_path: Path,
    model_storage_class: Type[ModelStorage],
    graph_runner_class: Type[GraphRunner],
) -> Tuple[ModelMetadata, GraphRunner]:
    """Loads a model from an archive and creates the prediction graph runner.

    Args:
        storage_path: Directory which contains the persisted graph components.
        model_archive_path: The path to the model archive.
        model_storage_class: The class to instantiate the model storage from.
        graph_runner_class: The class to instantiate the runner from.

    Returns:
        A tuple containing the model metadata and the prediction graph runner.
    """
    model_storage, model_metadata = model_storage_class.from_model_archive(
        storage_path=storage_path, model_archive_path=model_archive_path
    )
    runner = graph_runner_class.create(
        graph_schema=model_metadata.predict_schema,
        model_storage=model_storage,
        execution_context=ExecutionContext(
            graph_schema=model_metadata.predict_schema, model_id=model_metadata.model_id
        ),
    )
    return model_metadata, runner
