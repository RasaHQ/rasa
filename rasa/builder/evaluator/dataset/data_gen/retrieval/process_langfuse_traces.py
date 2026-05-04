"""CLI to build a JSONL dataset from Langfuse traces.

Usage:
    python -m rasa.builder.evaluator.scripts.process_langfuse_traces \\
        --trace-name <name> \\
        --kind {classification,retrieval} \\
        --output-dir <dir> \\
        --output-filename <file.jsonl>
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from rasa.builder.evaluator.artifacts import JSONLArtifact
from rasa.builder.evaluator.results_export import ResultsExporter
from rasa.builder.evaluator.tasks.base import AvailableTasks

structlogger = structlog.get_logger()


class TraceQuery(BaseModel):
    query: str
    source: str = "production"
    category: Optional[str] = "general"


def fetch_traces(
    *,
    trace_name: str,
    kind: AvailableTasks,
) -> list[Any]:
    """Fetch Langfuse traces by name with an annotation score match.

    Returns traces named ``trace_name`` that have at least one score with
    ``name == "eval"`` and ``value == eval_score_value`` within the last
    24 hours.
    """
    from rasa.builder.telemetry.langfuse_integration.langfuse_compat import langfuse

    client = langfuse.get_client()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=3)

    score_name = "eval"
    page_limit = 100

    matching_trace_ids: set[str] = set()
    page = 1
    while True:
        score_resp = client.api.score_v_2.get(
            name=score_name,
            from_timestamp=start_date,
            to_timestamp=end_date,
            page=page,
            limit=page_limit,
        )
        scores = score_resp.data or []
        for score in scores:
            trace_id = getattr(score, "trace_id", None)
            string_value = getattr(score, "string_value", None)
            if trace_id and string_value == kind.value:
                matching_trace_ids.add(trace_id)
        if len(scores) < page_limit:
            break
        page += 1

    structlogger.info(
        "fetch_traces.scores_matched",
        trace_id_count=len(matching_trace_ids),
    )

    traces: list[Any] = []
    for trace_id in matching_trace_ids:
        try:
            trace = client.api.trace.get(trace_id)
        except Exception as e:
            structlogger.warning(
                "fetch_traces.trace_get_failed",
                trace_id=trace_id,
                error=str(e),
            )
            continue
        if trace.name == trace_name:
            traces.append(trace)

    structlogger.info(
        "fetch_traces.done",
        trace_name=trace_name,
        matched=len(traces),
    )
    return traces


def _find_observation(trace: Any, name: str) -> Any:
    for obs in trace.observations or []:
        if obs.name == name:
            return obs
    raise ValueError(f"Observation '{name}' not found in trace {trace.id}")


def _process_classifier_trace(trace: Any) -> TraceQuery:
    obs = _find_observation(trace, "MessageClassifier.classify")
    return TraceQuery(query=obs.input["user_message"])


def _process_retrieval_trace(trace: Any) -> TraceQuery:
    obs = _find_observation(trace, "mcp_tool.search_rasa_documentation")
    return TraceQuery(query=obs.input["args"][1]["query"])


def process_trace(trace: Any, kind: AvailableTasks) -> TraceQuery:
    """Route a trace to the processor matching ``kind``."""
    if kind == AvailableTasks.CLASSIFICATION:
        return _process_classifier_trace(trace)
    if kind == AvailableTasks.RETRIEVAL:
        return _process_retrieval_trace(trace)
    raise ValueError(f"Unsupported trace kind: {kind}")


def build_dataset_from_traces(
    *,
    trace_name: str,
    kind: AvailableTasks,
    output_dir: Path,
    output_filename: str,
) -> None:
    """Fetch matching traces, process them, and write a JSONL dataset."""
    traces = fetch_traces(
        trace_name=trace_name,
        kind=kind,
    )
    records: list[TraceQuery] = []
    for trace in traces:
        try:
            records.append(process_trace(trace, kind))
        except (ValueError, KeyError) as e:
            structlogger.warning(
                "build_dataset_from_traces.skip_trace",
                trace_id=trace.id,
                error=str(e),
            )

    artifact = JSONLArtifact(filename=output_filename, records=records)
    ResultsExporter(output_dir=output_dir).export([artifact])

    structlogger.info(
        "build_dataset_from_traces.done",
        trace_name=trace_name,
        kind=kind,
        record_count=len(records),
        output=str(output_dir / output_filename),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-name", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=[t.value for t in AvailableTasks],
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-filename", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset_from_traces(
        trace_name=args.trace_name,
        kind=AvailableTasks(args.kind),
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
