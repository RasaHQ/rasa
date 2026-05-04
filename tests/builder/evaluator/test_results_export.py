"""Tests for ResultsFileWriter and ResultsExporter."""

import csv
import json
from unittest.mock import MagicMock

import yaml
from pydantic import BaseModel

from rasa.builder.evaluator.artifacts import (
    CSVArtifact,
    JSONLArtifact,
    YAMLArtifact,
)
from rasa.builder.evaluator.results_export import (
    ResultsExporter,
    ResultsFileWriter,
)


class _DummyRecord(BaseModel):
    name: str
    value: int


class TestResultsFileWriterYaml:
    def test_writes_data(self, tmp_path):
        writer = ResultsFileWriter(output_dir=tmp_path)

        writer.write_yaml(
            filename="20260408_120000_run_results.yaml",
            data={
                "experiment": {"run_id": "run-123"},
                "metrics": {"accuracy": 0.95},
            },
        )

        output_file = tmp_path / "20260408_120000_run_results.yaml"
        assert output_file.exists()
        with open(output_file) as f:
            data = yaml.safe_load(f)
        assert data["metrics"]["accuracy"] == 0.95
        assert data["experiment"]["run_id"] == "run-123"

    def test_empty_data(self, tmp_path):
        writer = ResultsFileWriter(output_dir=tmp_path)

        writer.write_yaml(filename="empty.yaml", data={})

        output_file = tmp_path / "empty.yaml"
        with open(output_file) as f:
            data = yaml.safe_load(f)
        assert data in ({}, None)

    def test_write_failure_does_not_raise(self, tmp_path):
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        writer = ResultsFileWriter(output_dir=readonly_dir)
        writer.write_yaml(filename="x.yaml", data={"k": "v"})
        readonly_dir.chmod(0o755)


class TestResultsFileWriterCsv:
    def test_writes_rows(self, tmp_path):
        writer = ResultsFileWriter(output_dir=tmp_path)
        rows = [
            {"a": "1", "b": "x"},
            {"a": "2", "b": "y"},
        ]

        writer.write_csv(filename="out.csv", fieldnames=["a", "b"], rows=rows)

        output_file = tmp_path / "out.csv"
        assert output_file.exists()
        with open(output_file) as f:
            read_rows = list(csv.DictReader(f))
        assert read_rows == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]

    def test_empty_rows_still_writes_header(self, tmp_path):
        writer = ResultsFileWriter(output_dir=tmp_path)
        writer.write_csv(filename="empty.csv", fieldnames=["a", "b"], rows=[])

        output_file = tmp_path / "empty.csv"
        assert output_file.exists()
        with open(output_file) as f:
            assert f.read().strip() == "a,b"

    def test_write_failure_does_not_raise(self, tmp_path):
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        writer = ResultsFileWriter(output_dir=readonly_dir)
        writer.write_csv(filename="x.csv", fieldnames=["a"], rows=[{"a": "1"}])
        readonly_dir.chmod(0o755)


class TestResultsFileWriterJsonl:
    def test_writes_records(self, tmp_path):
        writer = ResultsFileWriter(output_dir=tmp_path)
        records = [
            _DummyRecord(name="a", value=1),
            _DummyRecord(name="b", value=2),
        ]

        writer.write_jsonl(filename="out.jsonl", records=records)

        output_file = tmp_path / "out.jsonl"
        assert output_file.exists()
        lines = output_file.read_text().splitlines()
        assert [json.loads(line) for line in lines] == [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
        ]

    def test_empty_records_creates_empty_file(self, tmp_path):
        writer = ResultsFileWriter(output_dir=tmp_path)
        writer.write_jsonl(filename="empty.jsonl", records=[])

        output_file = tmp_path / "empty.jsonl"
        assert output_file.exists()
        assert output_file.read_text() == ""

    def test_creates_parent_directories(self, tmp_path):
        writer = ResultsFileWriter(output_dir=tmp_path / "nested" / "dir")
        writer.write_jsonl(
            filename="out.jsonl", records=[_DummyRecord(name="a", value=1)]
        )

        assert (tmp_path / "nested" / "dir" / "out.jsonl").exists()

    def test_write_failure_does_not_raise(self, tmp_path):
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        writer = ResultsFileWriter(output_dir=readonly_dir)
        writer.write_jsonl(
            filename="x.jsonl", records=[_DummyRecord(name="a", value=1)]
        )
        readonly_dir.chmod(0o755)


class TestResultsExporter:
    def test_dispatches_csv_artifact(self, tmp_path):
        exporter = ResultsExporter(output_dir=tmp_path)
        exporter._writer = MagicMock()

        artifact = CSVArtifact(
            filename="out.csv",
            fieldnames=["a", "b"],
            rows=[{"a": "1", "b": "2"}],
        )
        exporter.export(artifacts=[artifact])

        exporter._writer.write_csv.assert_called_once_with(
            filename="out.csv",
            fieldnames=["a", "b"],
            rows=[{"a": "1", "b": "2"}],
        )
        exporter._writer.write_yaml.assert_not_called()

    def test_dispatches_yaml_artifact(self, tmp_path):
        exporter = ResultsExporter(output_dir=tmp_path)
        exporter._writer = MagicMock()

        artifact = YAMLArtifact(
            filename="run.yaml",
            data={"metrics": {"accuracy": 0.9}},
        )
        exporter.export(artifacts=[artifact])

        exporter._writer.write_yaml.assert_called_once_with(
            filename="run.yaml",
            data={"metrics": {"accuracy": 0.9}},
        )
        exporter._writer.write_csv.assert_not_called()

    def test_dispatches_jsonl_artifact(self, tmp_path):
        exporter = ResultsExporter(output_dir=tmp_path)
        exporter._writer = MagicMock()

        records = [_DummyRecord(name="a", value=1)]
        artifact = JSONLArtifact(filename="out.jsonl", records=records)
        exporter.export(artifacts=[artifact])

        exporter._writer.write_jsonl.assert_called_once_with(
            filename="out.jsonl",
            records=records,
        )
        exporter._writer.write_csv.assert_not_called()
        exporter._writer.write_yaml.assert_not_called()

    def test_dispatches_mixed_artifact_list(self, tmp_path):
        exporter = ResultsExporter(output_dir=tmp_path)
        exporter._writer = MagicMock()

        artifacts = [
            CSVArtifact(filename="a.csv", fieldnames=["x"], rows=[]),
            YAMLArtifact(filename="b.yaml", data={"k": "v"}),
            JSONLArtifact(
                filename="c.jsonl",
                records=[_DummyRecord(name="a", value=1)],
            ),
        ]
        exporter.export(artifacts=artifacts)

        exporter._writer.write_csv.assert_called_once()
        exporter._writer.write_yaml.assert_called_once()
        exporter._writer.write_jsonl.assert_called_once()

    def test_empty_artifacts_list_noop(self, tmp_path):
        exporter = ResultsExporter(output_dir=tmp_path)
        exporter._writer = MagicMock()

        exporter.export(artifacts=[])

        exporter._writer.write_csv.assert_not_called()
        exporter._writer.write_yaml.assert_not_called()
