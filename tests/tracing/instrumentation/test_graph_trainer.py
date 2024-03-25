from pathlib import Path
from typing import Sequence, Text

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rasa.shared.data import TrainingType
from rasa.shared.importers.rasa import RasaFileImporter

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import (
    MockGraphTrainer,
    model_configuration,
)


@pytest.mark.parametrize(
    "training_type",
    [
        TrainingType.BOTH,
        TrainingType.NLU,
        TrainingType.CORE,
        TrainingType.END_TO_END,
    ],
)
async def test_tracing_for_training_without_finetuning(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    graph_trainer: MockGraphTrainer,
    tmp_path: Path,
    config_path: Text,
    domain_path: Text,
    data_path: Text,
    training_type: TrainingType,
    previous_num_captured_spans: int,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        graph_trainer_class=MockGraphTrainer,
    )

    expected_training_type = training_type.model_type
    expected_language = "en"
    expected_recipe_name = "graph.v1"
    expected_output_filename = "test_model.tar.gz"

    graph_model_config = model_configuration(config_path, training_type)
    importer = RasaFileImporter(
        config_file=config_path, domain_path=domain_path, training_data_paths=data_path
    )
    output_filename = tmp_path / "test_model.tar.gz"

    await graph_trainer.train(
        graph_model_config,
        importer,
        output_filename,
        is_finetuning=False,
    )

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockGraphTrainer.train"
    assert captured_span.attributes == {
        "training_type": expected_training_type,
        "language": expected_language,
        "recipe_name": expected_recipe_name,
        "output_filename": expected_output_filename,
        "is_finetuning": False,
    }


@pytest.mark.parametrize(
    "training_type",
    [
        TrainingType.BOTH,
        TrainingType.NLU,
        TrainingType.CORE,
        TrainingType.END_TO_END,
    ],
)
async def test_tracing_for_training_with_finetuning(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    graph_trainer: MockGraphTrainer,
    tmp_path: Path,
    config_path: Text,
    domain_path: Text,
    data_path: Text,
    training_type: TrainingType,
    previous_num_captured_spans: int,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        graph_trainer_class=MockGraphTrainer,
    )

    expected_training_type = training_type.model_type
    expected_language = "en"
    expected_recipe_name = "graph.v1"
    expected_output_filename = "test_model.tar.gz"

    graph_model_config = model_configuration(config_path, training_type)
    importer = RasaFileImporter(
        config_file=config_path, domain_path=domain_path, training_data_paths=data_path
    )
    output_filename = tmp_path / "test_model.tar.gz"

    # we need to train the model first without finetuning
    await graph_trainer.train(
        graph_model_config,
        importer,
        output_filename,
        is_finetuning=False,
    )

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockGraphTrainer.train"
    assert captured_span.attributes == {
        "training_type": expected_training_type,
        "language": expected_language,
        "recipe_name": expected_recipe_name,
        "output_filename": expected_output_filename,
        "is_finetuning": False,
    }

    # changing the constructor name of the finetuning validator node from `create`
    # to `load` is required in order to finetune the initial trained model
    finetuning_validator_schema_node = graph_model_config.train_schema.nodes.get(
        "finetuning_validator"
    )
    finetuning_validator_schema_node.constructor_name = "load"
    graph_model_config.train_schema.nodes[
        "finetuning_validator"
    ] = finetuning_validator_schema_node

    await graph_trainer.train(
        graph_model_config,
        importer,
        output_filename,
        is_finetuning=True,
    )

    finetuning_captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_finetuning_captured_spans = (
        len(finetuning_captured_spans)
        - num_captured_spans
        - previous_num_captured_spans
    )
    assert num_finetuning_captured_spans == 1

    captured_span = captured_spans[-1]

    finetuning_captured_span = finetuning_captured_spans[-1]
    assert finetuning_captured_span.name == "MockGraphTrainer.train"
    assert finetuning_captured_span.attributes == {
        "training_type": expected_training_type,
        "language": expected_language,
        "recipe_name": expected_recipe_name,
        "output_filename": expected_output_filename,
        "is_finetuning": True,
    }
