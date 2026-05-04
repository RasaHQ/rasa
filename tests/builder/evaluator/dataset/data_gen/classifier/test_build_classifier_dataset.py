"""End-to-end test of the classifier data-gen orchestrator.

Covers ``_run`` plus its pure helpers (``_load_queries``, ``_assemble_entries``,
``_generate_entry_id``) by mocking only the LLM/Langfuse boundaries.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rasa.builder.evaluator.configs.models import (
    ClassifierLabelingConfig,
    LabelingConfig,
)
from rasa.builder.evaluator.dataset.data_gen.classifier import (
    build_classifier_dataset as mod,
)
from rasa.builder.evaluator.dataset.data_gen.classifier.label_queries import (
    LabeledQuery,
    SkippedQuery,
)


def _make_config(tmp_path: Path) -> LabelingConfig:
    return LabelingConfig(
        dataset_name="ds",
        dataset_description="desc",
        output_dir=str(tmp_path / "out" / "classifier.jsonl"),
        queries_path=str(tmp_path / "queries.jsonl"),
        model="gpt-4o-mini",
        batch_size=10,
        batch_pause_seconds=0.0,
        classifier=ClassifierLabelingConfig(
            valid_categories=["copilot", "error_fallback"]
        ),
    )


def _write_queries(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


class TestRun:
    async def test_full_pipeline_skips_invalid_lines_and_pushes(self, tmp_path):
        queries_path = tmp_path / "queries.jsonl"
        _write_queries(
            queries_path,
            [
                json.dumps({"query": "How do I write a flow?", "source": "manual"}),
                "{not valid json",
                json.dumps({"no_query": "missing"}),
                json.dumps({"query": "What is Rasa Pro?"}),
                json.dumps({"query": "Trigger an error case"}),
            ],
        )

        output_path = tmp_path / "out" / "classifier.jsonl"
        config = _make_config(tmp_path)

        async def fake_label_queries(queries, llm_client, cfg):
            assert len(queries) == 3
            labeled = [
                (queries[0], LabeledQuery(category="copilot", confidence="high")),
                (queries[1], LabeledQuery(category="copilot", confidence="medium")),
            ]
            skipped = [SkippedQuery(query=queries[2]["query"], reason="boom")]
            return labeled, skipped

        with (
            patch.object(mod, "validate_env") as mock_validate,
            patch.object(mod, "AsyncOpenAI") as mock_openai,
            patch.object(mod, "label_queries", side_effect=fake_label_queries),
            patch.object(mod, "push_to_langfuse") as mock_push,
        ):
            rc = await mod._run(
                queries_path=queries_path,
                output_path=output_path,
                config=config,
                langfuse_dataset_name="ds-name",
                no_langfuse=False,
            )

        assert rc == 0
        mock_validate.assert_called_once_with(True)
        mock_openai.assert_called_once()
        mock_push.assert_called_once()
        pushed_entries, dataset_name, description = mock_push.call_args.args
        assert dataset_name == "ds-name"
        assert description == "desc"
        assert len(pushed_entries) == 2
        # IDs are deterministic + match _generate_entry_id
        assert pushed_entries[0].id == mod._generate_entry_id("How do I write a flow?")
        assert pushed_entries[0].id == mod._generate_entry_id(
            "  HOW DO I WRITE A FLOW?  "
        )
        # JSONL artifacts written for both labeled and skipped
        assert output_path.exists()
        assert (output_path.parent / "classifier_skipped.jsonl").exists()

        labeled_lines = output_path.read_text().strip().splitlines()
        assert len(labeled_lines) == 2
        first = json.loads(labeled_lines[0])
        assert first["expected_output"]["response_category"] == "copilot"
        assert first["metadata"]["ids"]["source"] == "manual"

    async def test_no_queries_returns_error(self, tmp_path):
        queries_path = tmp_path / "queries.jsonl"
        queries_path.write_text("")
        output_path = tmp_path / "out" / "classifier.jsonl"
        config = _make_config(tmp_path)

        with (
            patch.object(mod, "validate_env"),
            patch.object(mod, "AsyncOpenAI"),
            patch.object(mod, "label_queries", new=AsyncMock()) as mock_label,
            patch.object(mod, "push_to_langfuse") as mock_push,
        ):
            rc = await mod._run(
                queries_path=queries_path,
                output_path=output_path,
                config=config,
                langfuse_dataset_name=None,
                no_langfuse=True,
            )

        assert rc == 1
        mock_label.assert_not_called()
        mock_push.assert_not_called()

    async def test_no_langfuse_skips_push(self, tmp_path):
        queries_path = tmp_path / "queries.jsonl"
        _write_queries(queries_path, [json.dumps({"query": "q1", "source": "manual"})])
        output_path = tmp_path / "out" / "classifier.jsonl"
        config = _make_config(tmp_path)

        async def fake_label_queries(queries, llm_client, cfg):
            return (
                [(queries[0], LabeledQuery(category="copilot", confidence="low"))],
                [],
            )

        with (
            patch.object(mod, "validate_env") as mock_validate,
            patch.object(mod, "AsyncOpenAI"),
            patch.object(mod, "label_queries", side_effect=fake_label_queries),
            patch.object(mod, "push_to_langfuse") as mock_push,
        ):
            rc = await mod._run(
                queries_path=queries_path,
                output_path=output_path,
                config=config,
                langfuse_dataset_name="ds",
                no_langfuse=True,
            )

        assert rc == 0
        mock_validate.assert_called_once_with(False)
        mock_push.assert_not_called()
        # No skipped artifact when there are no skipped queries
        assert not (output_path.parent / "classifier_skipped.jsonl").exists()


def test_generate_entry_id_is_deterministic_and_normalized():
    a = mod._generate_entry_id("Hello world")
    b = mod._generate_entry_id("  hello WORLD  ")
    assert a == b
    assert a.startswith("classifier_")
