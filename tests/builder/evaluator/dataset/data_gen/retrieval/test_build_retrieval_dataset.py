"""End-to-end test of the retrieval data-gen orchestrator.

Covers ``_run`` plus ``docs_index.build_index`` (parsing, frontmatter,
markdown cleaning, URL derivation, JSONL round-trip) and ``_load_queries`` /
``_assemble_entries``. Mocks LLM/Langfuse boundaries.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from rasa.builder.evaluator.configs.models import (
    LabelingConfig,
    RetrievalLabelingConfig,
)
from rasa.builder.evaluator.dataset.data_gen.retrieval import (
    build_retrieval_dataset as mod,
)
from rasa.builder.evaluator.dataset.data_gen.retrieval import (
    docs_index as docs_index_mod,
)
from rasa.builder.evaluator.dataset.retrieval_models import RelevantPage
from rasa.builder.evaluator.tasks.base import AvailableTasks


def _make_config(tmp_path: Path) -> LabelingConfig:
    return LabelingConfig(
        dataset_name="ds",
        dataset_description="desc",
        output_dir=str(tmp_path / "out" / "retrieval.jsonl"),
        queries_path=str(tmp_path / "queries.jsonl"),
        model="gpt-4o-mini",
        batch_size=10,
        batch_pause_seconds=0.0,
        retrieval=RetrievalLabelingConfig(docs_repo_path=str(tmp_path / "repo")),
    )


def _build_docs_repo(repo_root: Path) -> None:
    docs_dir = repo_root / "docs"
    pages_dir = docs_dir / "pro" / "build"
    pages_dir.mkdir(parents=True)
    (pages_dir / "writing-flows.mdx").write_text(
        "---\n"
        "title: Writing Flows\n"
        "id: flows-page\n"
        "---\n"
        "import Foo from 'foo';\n"
        "<Bar />\n"
        "Hello <Inline>world</Inline>.\n"
    )
    # Plain page without frontmatter
    plain = docs_dir / "intro.mdx"
    plain.write_text("Just plain content here.\n")
    # Excluded dir — must be skipped
    snippets = docs_dir / "snippets"
    snippets.mkdir()
    (snippets / "ignored.mdx").write_text("---\ntitle: Ignored\n---\nshould skip\n")
    # Empty page after cleaning — must be skipped with warning
    empty = docs_dir / "empty.mdx"
    empty.write_text("---\ntitle: Empty\n---\n")


class TestRun:
    async def test_full_pipeline(self, tmp_path):
        repo_root = tmp_path / "repo"
        _build_docs_repo(repo_root)

        queries_path = tmp_path / "queries.jsonl"
        queries_path.write_text(
            json.dumps({"query": "How do I write flows?"})
            + "\n"
            + "{bad json line\n"
            + json.dumps(
                {"query": "What is intro?", "category": "concept", "source": "langfuse"}
            )
            + "\n"
        )

        output_path = tmp_path / "out" / "retrieval.jsonl"
        config = _make_config(tmp_path)

        async def fake_label_queries(queries, doc_index, llm_client, cfg):
            # Confirm the index was built and passed in
            assert {p.doc_id for p in doc_index.pages} == {
                "pro/build/flows-page",
                "intro",
            }
            page = next(
                p for p in doc_index.pages if p.doc_id == "pro/build/flows-page"
            )
            labeled = [
                (
                    queries[0],
                    [RelevantPage(doc_id=page.doc_id, url=page.url, confidence="high")],
                ),
                (queries[1], []),
            ]
            return labeled, []

        fake_run = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n")
        with (
            patch.object(mod, "validate_env") as mock_validate,
            patch.object(mod, "AsyncOpenAI"),
            patch.object(mod, "label_queries", side_effect=fake_label_queries),
            patch.object(mod, "push_to_langfuse") as mock_push,
            patch.object(docs_index_mod.subprocess, "run", return_value=fake_run),
        ):
            rc = await mod._run(
                docs_repo_path=repo_root,
                queries_path=queries_path,
                output_path=output_path,
                config=config,
                langfuse_dataset_name="ds-name",
                no_langfuse=False,
            )

        assert rc == 0
        mock_validate.assert_called_once_with(True, AvailableTasks.RETRIEVAL)
        mock_push.assert_called_once()
        pushed_entries, dataset_name, description = mock_push.call_args.args
        assert dataset_name == "ds-name"
        assert description == "desc"
        assert len(pushed_entries) == 2
        # Deterministic IDs and metadata commit propagated from the docs repo
        assert pushed_entries[0].id == mod._generate_entry_id("How do I write flows?")
        assert pushed_entries[0].metadata.docs_repo_commit == "abc123"
        # Defaults applied to the second query
        assert pushed_entries[1].metadata.category == "concept"
        assert pushed_entries[1].metadata.source == "langfuse"
        # Output JSONL written, no skipped artifact
        assert output_path.exists()
        assert not (output_path.parent / "retrieval_skipped.jsonl").exists()

        first = json.loads(output_path.read_text().splitlines()[0])
        assert (
            first["expected_output"]["relevant_pages"][0]["doc_id"]
            == "pro/build/flows-page"
        )
        assert first["expected_output"]["relevant_pages"][0]["url"].endswith(
            "/pro/build/flows-page"
        )

    async def test_no_queries_returns_error(self, tmp_path):
        repo_root = tmp_path / "repo"
        _build_docs_repo(repo_root)
        queries_path = tmp_path / "queries.jsonl"
        queries_path.write_text("")
        output_path = tmp_path / "out" / "retrieval.jsonl"
        config = _make_config(tmp_path)

        with (
            patch.object(mod, "validate_env"),
            patch.object(mod, "AsyncOpenAI"),
            patch.object(mod, "label_queries") as mock_label,
            patch.object(mod, "push_to_langfuse") as mock_push,
            patch.object(
                docs_index_mod.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="x"
                ),
            ),
        ):
            rc = await mod._run(
                docs_repo_path=repo_root,
                queries_path=queries_path,
                output_path=output_path,
                config=config,
                langfuse_dataset_name=None,
                no_langfuse=True,
            )

        assert rc == 1
        mock_label.assert_not_called()
        mock_push.assert_not_called()


class TestDocsIndexHelpers:
    def test_clean_markdown_strips_imports_and_jsx(self):
        body = "import Foo from 'foo';\n<SelfClose />\n<Wrap>kept</Wrap>\n\n\n\nend\n"
        cleaned = docs_index_mod._clean_markdown(body)
        assert "import" not in cleaned
        assert "SelfClose" not in cleaned
        assert "kept" in cleaned
        assert "<Wrap>" not in cleaned
        # Excessive blank lines collapsed
        assert "\n\n\n" not in cleaned

    def test_parse_frontmatter_returns_empty_when_absent(self):
        fm, body = docs_index_mod._parse_frontmatter("no frontmatter here\n")
        assert fm == {}
        assert "no frontmatter" in body

    def test_derive_url_uses_frontmatter_id(self):
        url, doc_id = docs_index_mod._derive_url_and_doc_id(
            Path("pro/build/writing-flows.mdx"), frontmatter_id="flows"
        )
        assert doc_id == "pro/build/flows"
        assert url.endswith("/pro/build/flows")

    def test_derive_url_root_level(self):
        url, doc_id = docs_index_mod._derive_url_and_doc_id(
            Path("intro.mdx"), frontmatter_id=None
        )
        assert doc_id == "intro"
        assert url.endswith("/intro")

    def test_save_and_load_jsonl_round_trip(self, tmp_path):
        repo_root = tmp_path / "repo"
        _build_docs_repo(repo_root)
        fake_run = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="deadbeef\n"
        )
        with patch.object(docs_index_mod.subprocess, "run", return_value=fake_run):
            index = docs_index_mod.build_index(repo_root)

        assert index.docs_repo_commit == "deadbeef"
        assert {p.doc_id for p in index.pages} == {"pro/build/flows-page", "intro"}

        out = tmp_path / "index.jsonl"
        index.save_to_jsonl(out)
        reloaded = docs_index_mod.DocIndex.load_from_jsonl(out)
        assert reloaded.docs_repo_commit == index.docs_repo_commit
        assert [p.model_dump() for p in reloaded.pages] == [
            p.model_dump() for p in index.pages
        ]

    def test_get_git_commit_returns_unknown_on_failure(self, tmp_path):
        with patch.object(
            docs_index_mod.subprocess,
            "run",
            side_effect=FileNotFoundError(),
        ):
            assert docs_index_mod._get_git_commit(tmp_path) == "unknown"


def test_generate_entry_id_is_deterministic_and_normalized():
    a = mod._generate_entry_id("Hello world")
    b = mod._generate_entry_id("  hello WORLD  ")
    assert a == b
    assert a.startswith("retrieval_")
