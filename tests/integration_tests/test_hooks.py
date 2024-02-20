from pathlib import Path

from rasa.plugin import plugin_manager

from rasa.anonymization.anonymization_pipeline import (
    BackgroundAnonymizationPipeline,
    SyncAnonymizationPipeline,
)


def test_get_anonymization_pipeline_no_endpoints() -> None:
    plugin_manager().hook.init_anonymization_pipeline(endpoints_file=None)
    pipeline = plugin_manager().hook.get_anonymization_pipeline()
    assert pipeline is None


def test_get_anonymization_pipeline() -> None:
    endpoints_file = (
        Path(__file__).parent.parent.parent / "data" / "anonymization" / "endpoints.yml"
    )

    # first load the anonymization pipeline
    plugin_manager().hook.init_anonymization_pipeline(endpoints_file=endpoints_file)

    pipeline = plugin_manager().hook.get_anonymization_pipeline()

    assert isinstance(pipeline, BackgroundAnonymizationPipeline)
    assert isinstance(pipeline.anonymization_pipeline, SyncAnonymizationPipeline)
    assert len(pipeline.anonymization_pipeline.orchestrators) == 2

    pipeline.stop()
