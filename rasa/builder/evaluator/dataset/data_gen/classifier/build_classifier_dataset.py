"""Build a classifier evaluation dataset: label queries and push to Langfuse.

Orchestrates the full pipeline:
1. Load seed queries from a JSONL file.
2. Label each query with a single ResponseCategory using LLM-as-judge.
3. Assemble DatasetEntry objects.
4. Push to Langfuse (optional) and save locally as JSONL.

Usage:
    Module: rasa.builder.evaluator.dataset.data_gen.classifier.build_classifier_dataset
    Config: rasa/builder/evaluator/configs/build_classifier_dataset.yaml
    Run: uv run python -m <module> --config <config>

Required env vars: OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
"""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import structlog
from openai import AsyncOpenAI

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.artifacts import JSONLArtifact
from rasa.builder.evaluator.configs.models import (
    ConfigType,
    LabelingConfig,
    load_config,
)
from rasa.builder.evaluator.dataset.classifier_models import (
    DatasetEntry,
    DatasetExpectedOutput,
    DatasetInput,
    DatasetMetadata,
)
from rasa.builder.evaluator.dataset.data_gen.classifier.label_queries import (
    LabeledQuery,
    label_queries,
)
from rasa.builder.evaluator.helpers import (
    push_to_langfuse,
    validate_env,
)
from rasa.builder.evaluator.results_export import ResultsExporter

structlogger = structlog.get_logger()


def _generate_entry_id(query: str) -> str:
    """Generate a deterministic ID from the query text."""
    query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:12]
    return f"classifier_{query_hash}"


def _load_queries(queries_path: Path) -> list[dict[str, str]]:
    """Load seed queries from a JSONL file.

    Each line must be a JSON object with at least a ``query`` key.
    Optional keys: ``source`` (default: ``manual``).
    """
    queries: list[dict[str, str]] = []
    with open(queries_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                structlogger.warning(
                    "build_dataset.load_queries.invalid_line",
                    line_num=line_num,
                    error=str(e),
                )
                continue

            if "query" not in entry:
                structlogger.warning(
                    "build_dataset.load_queries.missing_query",
                    line_num=line_num,
                )
                continue

            entry.setdefault("source", "manual")
            queries.append(entry)

    structlogger.info(
        "build_dataset.load_queries.done",
        path=str(queries_path),
        count=len(queries),
    )
    return queries


def _assemble_entries(
    labeled: list[tuple[dict[str, str], LabeledQuery]],
) -> list[DatasetEntry]:
    """Convert labeled query results into DatasetEntry objects."""
    entries: list[DatasetEntry] = []

    for query_dict, judgment in labeled:
        query_text = query_dict["query"]
        entry = DatasetEntry(
            id=_generate_entry_id(query_text),
            input=DatasetInput(message=query_text),
            expected_output=DatasetExpectedOutput(
                answer="",
                response_category=ResponseCategory(judgment.category),
                references=[],
            ),
            metadata=DatasetMetadata(
                ids={
                    "source": query_dict.get("source", "manual"),
                    "confidence": judgment.confidence,
                },
            ),
        )
        entries.append(entry)

    return entries


async def _run(
    queries_path: Path,
    output_path: Path,
    config: LabelingConfig,
    langfuse_dataset_name: Optional[str],
    no_langfuse: bool = False,
) -> int:
    push_langfuse = langfuse_dataset_name is not None and not no_langfuse
    validate_env(push_langfuse)

    structlogger.info("build_dataset.step", step="load_queries")
    queries = _load_queries(queries_path)
    if not queries:
        structlogger.error("build_dataset.no_queries")
        return 1

    structlogger.info("build_dataset.step", step="label_queries")
    llm_client = AsyncOpenAI()
    labeled, skipped = await label_queries(queries, llm_client, config)

    structlogger.info("build_dataset.step", step="assemble_entries")
    entries = _assemble_entries(labeled)

    structlogger.info("build_dataset.step", step="save_local")
    artifacts: list = [
        JSONLArtifact(filename=output_path.name, records=list(entries)),
    ]
    if skipped:
        skipped_filename = output_path.stem + "_skipped" + output_path.suffix
        artifacts.append(
            JSONLArtifact(filename=skipped_filename, records=list(skipped))
        )
    ResultsExporter(output_dir=output_path.parent).export(artifacts)

    if push_langfuse:
        push_to_langfuse(entries, langfuse_dataset_name, config.dataset_description)

    structlogger.info(
        "build_dataset.done",
        total_queries=len(queries),
        labeled=len(entries),
        skipped=len(skipped),
        output=str(output_path),
        langfuse_dataset=langfuse_dataset_name,
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a classifier evaluation dataset with LLM-based labeling.",
    )
    parser.add_argument(
        "--config",
        required=True,
        default="rasa/builder/evaluator/configs/build_classifier_dataset.yaml",
        help="Path to the labeling YAML config file.",
    )
    parser.add_argument(
        "--no-langfuse-push",
        action="store_true",
        help="Skip pushing results to Langfuse.",
    )
    args = parser.parse_args()

    config = load_config(args.config, ConfigType.LABELING)

    return asyncio.run(
        _run(
            queries_path=Path(config.queries_path).resolve(),
            output_path=Path(config.output_dir).resolve(),
            config=config,
            langfuse_dataset_name=config.dataset_name,
            no_langfuse=args.no_langfus_push,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
