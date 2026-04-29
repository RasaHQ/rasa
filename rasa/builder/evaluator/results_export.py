"""Results export — writes experiment artifacts to disk.

Two classes:

- ``ResultsFileWriter``: generic file I/O primitives (YAML, CSV).
- ``ResultsExporter``: format-dispatcher. It iterates an ``Artifact`` list
  and routes each one to the appropriate writer method based on type.

Keeps the runner focused on coordination and evaluators focused on data,
while giving persistence its own layer.
"""

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import structlog
import yaml  # type: ignore[import-untyped]

from rasa.builder.evaluator.artifacts import Artifact, CSVArtifact, YAMLArtifact

structlogger = structlog.get_logger()


class ResultsFileWriter:
    """Generic file I/O primitives for experiment artifacts.

    Methods never raise; failures are logged and skipped so a best-effort
    export never aborts an experiment run.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write_yaml(self, filename: str, data: Mapping[str, Any]) -> None:
        """Write ``data`` as YAML under the configured output dir."""
        output_path = self._output_dir / filename
        try:
            with open(str(output_path), "w") as f:
                yaml.dump(dict(data), f, default_flow_style=False, sort_keys=False)
            structlogger.info("results_export.yaml", file=str(output_path))
        except Exception as e:
            structlogger.error(
                "results_export.yaml_failed",
                file=str(output_path),
                error=str(e),
            )

    def write_csv(
        self,
        filename: str,
        fieldnames: Sequence[str],
        rows: Iterable[Mapping[str, Any]],
    ) -> None:
        """Write ``rows`` as a CSV file under the configured output dir.

        Args:
            filename: File name (no directory component).
            fieldnames: Column names in order.
            rows: Iterable of dicts; keys should be a subset of ``fieldnames``.
        """
        output_path = self._output_dir / filename
        try:
            rows_list = list(rows)
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(fieldnames))
                writer.writeheader()
                writer.writerows(rows_list)
            structlogger.info(
                "results_export.csv",
                file=str(output_path),
                count=len(rows_list),
            )
        except Exception as e:
            structlogger.error(
                "results_export.csv_failed",
                file=str(output_path),
                error=str(e),
            )


class ResultsExporter:
    """Dispatches ``Artifact`` objects to the appropriate writer method.

    The exporter has no knowledge of evaluators, experiment results, or
    task semantics — it only looks at artifact type and routes.
    """

    def __init__(self, output_dir: Path) -> None:
        self._writer = ResultsFileWriter(output_dir)

    def export(self, artifacts: Sequence[Artifact]) -> None:
        """Write each artifact using the writer method matching its type."""
        for artifact in artifacts:
            if isinstance(artifact, CSVArtifact):
                self._writer.write_csv(
                    filename=artifact.filename,
                    fieldnames=artifact.fieldnames,
                    rows=artifact.rows,
                )
            elif isinstance(artifact, YAMLArtifact):
                self._writer.write_yaml(
                    filename=artifact.filename,
                    data=artifact.data,
                )
            else:
                structlogger.warning(
                    "results_export.unknown_artifact",
                    artifact_type=type(artifact).__name__,
                )
