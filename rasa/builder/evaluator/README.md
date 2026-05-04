# Evaluator

Offline evaluation framework for Copilot components. Experiments run against
[Langfuse](https://langfuse.com) datasets, metrics are tracked in Langfuse and
exported locally as YAML. The framework is config-driven — switching between
experiments requires only a YAML file change.

## Architecture

```
┌────────────────┐
│  YAML Config   │   name, task, dataset, output formats
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ ExperimentRunner│   Resolves task + evaluator from registry,
│   (runner.py)  │   orchestrates the Langfuse experiment
└───────┬────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌────────────────┐        ┌────────────────┐
│     Task       │        │   Evaluator    │
│  (BaseTask)    │        │ (BaseEvaluator)│
│                │        │                │
│  Called once   │        │  run-level:    │
│  per dataset   │        │   called once  │
│  item          │        │   after all    │
│                │        │   items        │
│  Returns a     │        │  item-level:   │
│  task-specific │        │   called per   │
│  result model  │        │   item         │
└────────────────┘        └────────────────┘
        │                          │
        ▼                          ▼
┌──────────────────────────────────────────┐
│              Langfuse                    │
│  Dataset run + Evaluation objects        │
│  + local export (YAML, CSV)              │
└──────────────────────────────────────────┘
```

The framework has four layers:

### Config (`configs/`)

Each experiment is defined by a YAML file validated against `ExperimentConfig`
in `configs/models.py`:

| Field          | Description                                       |
|----------------|---------------------------------------------------|
| `name`         | Experiment name (shown in Langfuse)               |
| `description`  | Human-readable description                        |
| `task`         | Task type (available: `classification`, `retrieval`) |
| `dataset_name` | Langfuse dataset to evaluate against              |
| `results_dir`  | Local directory for exported results              |
| `formats`      | Output formats: `langfuse`, `yaml`                |

### Runner (`runner.py`)

`ExperimentRunner` is the single entry point. On init it:

1. Loads and validates the YAML config.
2. Looks up the task + evaluator pair from the **registry** — a dict mapping
   `AvailableTasks` to `EvaluatorEntry(task_cls, eval_cls, level)`.
3. Fetches the dataset from Langfuse.

On `run_experiment()` it instantiates the task and evaluator, then calls
Langfuse's `dataset.run_experiment()` with the correct evaluator key:

- `level=RUN` → `run_evaluators` (batch — called once after all items)
- `level=ITEM` → `evaluators` (per-item — called for each item)

### Tasks (`tasks/`)

A task is a class extending `BaseTask`. It receives a single Langfuse dataset
item and returns a task-specific result model (e.g. `ClassifierTaskResult`,
`RetrievalTaskResult`):

```python
class BaseTask(ABC):
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    @abstractmethod
    async def run_task(self, *, item: Any, **kwargs: Any) -> Optional[Any]: ...
```

- `__init__` — receives the full `ExperimentConfig` so the task can read
  task-specific fields (e.g. `config.retrieval`); set up expensive resources once
- `run_task` — called per dataset item by Langfuse

### Evaluators (`evaluators/`)

An evaluator extends `BaseEvaluator` which provides a template method (`run`)
that orchestrates three abstract steps:

```
extract_results(item_results) → evaluate(results) → to_evaluations(summary, skip_count)
```

- `extract_results` — parse Langfuse `ExperimentItemResult` objects into
  domain-specific data, skip invalid items
- `evaluate` — compute metrics from parsed results
- `to_evaluations` — convert metrics into Langfuse `Evaluation` objects

The `run()` method is the Langfuse entry point — subclasses should not override it.

## Adding a new experiment

1. **Create a task** in `tasks/`:
   - Extend `BaseTask`
   - Define a task-specific result model in `tasks/base.py`
   - Initialize resources in `__init__(config)` and call `super().__init__(config)`
   - Implement `run_task(*, item, **kwargs) -> Optional[<YourResultModel>]`

2. **Create an evaluator** in `evaluators/<name>/`:
   - Extend `BaseEvaluator`
   - Implement `extract_results`, `evaluate`, `to_evaluations`

3. **Register** in `runner.py`:
   - Add a member to `AvailableTasks` enum
   - Add an entry to `_build_evaluator_registry()`:
     ```python
     AvailableTasks.MY_TASK: EvaluatorEntry(
         task_cls=MyTask,
         eval_cls=MyEvaluator,
         level=AvailableLevels.RUN,  # or ITEM
     ),
     ```

4. **Update config schema** (if needed):
   - Add the new task value to the `Literal` in `ExperimentConfig.task`
     (`configs/models.py`)

5. **Create a YAML config** in `configs/`:
   ```yaml
   name: my_experiment
   description: >
     What this experiment evaluates.
   task: my_task
   dataset_name: my-langfuse-dataset
   results_dir: "results/my_experiment"
   formats: [langfuse, yaml]
   ```

6. **Run it**:
   ```bash
   python -m rasa.builder.evaluator.run_experiment \
       --config rasa/builder/evaluator/configs/my_config.yaml
   ```

## Existing experiments

### Message Classifier

Evaluates the `MessageClassifier` in isolation (no copilot response generation).
Reports precision, recall, and F1 per class and overall (micro, macro, weighted).

| Property    | Value                                                        |
|-------------|--------------------------------------------------------------|
| Config      | `configs/test_message_classifier.yaml`                        |
| Task        | `ClassifierTask` (`tasks/classifier_task.py`)                 |
| Evaluator   | `ClassificationEvaluator` (`evaluators/classification/`)      |
| Level       | `run` (batch metrics after all items)                         |
| Dataset     | `user-input-classification-dataset`                           |
| Metrics     | accuracy, precision, recall, F1 (micro/macro/weighted), per-class |

**Run:**

```bash
python -m rasa.builder.evaluator.run_experiment \
    --config rasa/builder/evaluator/configs/test_message_classifier.yaml
```

### Retrieval

Evaluates document retrieval quality in isolation (no copilot response generation).
Reports recall@K, MRR, latency, error/empty rates overall and per query category.

| Property    | Value                                                        |
|-------------|--------------------------------------------------------------|
| Config      | `configs/test_retrieval.yaml`                                |
| Task        | `RetrievalTask` (`tasks/retrieval_task.py`)                  |
| Evaluator   | `RetrievalEvaluator` (`evaluators/retrieval/`)               |
| Level       | `run` (batch metrics after all items)                        |
| Dataset     | `retrieval-eval-v1`                                          |
| Metrics     | recall@3/5/10, MRR, latency (mean/p50/p95), error rate, empty result rate, per-category breakdown |

**Run:**

```bash
python -m rasa.builder.evaluator.run_experiment \
    --config rasa/builder/evaluator/configs/test_retrieval.yaml
```

## Dataset generation

Before running an experiment you need a labeled dataset in Langfuse. The
`dataset/data_gen/` module provides config-driven pipelines to build classifier
and retrieval datasets from seed queries using LLM-as-judge labeling, then push
them to Langfuse and save a local JSONL copy.

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Seed queries    │───▶│  LLM labeling    │───▶│  Dataset entries │
│   (JSONL)        │    │  (gpt-4.1)       │    │  (JSONL + LF)    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

Seed queries can be hand-curated or extracted from production Langfuse traces
via [process_langfuse_traces.py](rasa/builder/evaluator/dataset/data_gen/retrieval/process_langfuse_traces.py).

### Classifier dataset

Single-stage labeling: each query is assigned one `ResponseCategory` label.

| Property      | Value                                                        |
|---------------|--------------------------------------------------------------|
| Builder       | [build_classifier_dataset.py](rasa/builder/evaluator/dataset/data_gen/classifier/build_classifier_dataset.py) |
| Config        | [configs/build_classifier_dataset.yaml](rasa/builder/evaluator/configs/build_classifier_dataset.yaml) |
| Input         | JSONL of seed queries                                        |
| Output        | Langfuse dataset + local JSONL of `DatasetEntry` records     |
| Model         | `gpt-4.1-2025-04-14` (configurable)                          |

**Run:**

```bash
uv run python -m rasa.builder.evaluator.dataset.data_gen.classifier.build_classifier_dataset \
    --config rasa/builder/evaluator/configs/build_classifier_dataset.yaml
```

### Retrieval dataset

Two-stage labeling: an index is built from the docs repo, then each query is
labeled with the set of relevant pages.

| Property      | Value                                                        |
|---------------|--------------------------------------------------------------|
| Builder       | [build_retrieval_dataset.py](rasa/builder/evaluator/dataset/data_gen/retrieval/build_retrieval_dataset.py) |
| Config        | [configs/label_retrieval_dataset.yaml](rasa/builder/evaluator/configs/label_retrieval_dataset.yaml) |
| Input         | JSONL of seed queries + path to docs repo                    |
| Output        | Langfuse dataset + local JSONL of `RetrievalDatasetEntry` records |
| Model         | `gpt-4.1-2025-04-14` (configurable)                          |

**Run:**

```bash
uv run python -m rasa.builder.evaluator.dataset.data_gen.retrieval.build_retrieval_dataset \
    --config rasa/builder/evaluator/configs/label_retrieval_dataset.yaml
```

### Common labeling config fields

Both pipelines share the `LabelingConfig` schema (`configs/models.py`):

| Field                   | Description                                          |
|-------------------------|------------------------------------------------------|
| `dataset_name`          | Target Langfuse dataset name                         |
| `dataset_description`   | Human-readable description                           |
| `queries_path`          | Path to seed queries JSONL                           |
| `output_dir`            | Local directory for the JSONL copy                   |
| `model` / `temperature` | LLM used for labeling                                |
| `batch_size` / `batch_pause_seconds` | Concurrency + rate-limit controls       |
| `max_retries`           | Retry budget per query                               |
| `classifier.valid_categories` | (classifier) Allowed labels                    |
| `retrieval.docs_repo_path`    | (retrieval) Path to the documentation repo     |

### Sourcing seed queries from Langfuse traces

To bootstrap seed queries from production traffic:

```bash
uv run python -m rasa.builder.evaluator.dataset.data_gen.retrieval.process_langfuse_traces \
    --trace-name {copilot} \
    --kind {classification,retrieval} \
    --output-dir <dir> \
    --output-filename <file.jsonl>
```

The resulting JSONL can be passed as `queries_path` to either builder above.

## Environment variables

| Variable              | Required | Description                          |
|-----------------------|----------|--------------------------------------|
| `OPENAI_API_KEY`      | Yes      | OpenAI API key for LLM calls         |
| `LANGFUSE_PUBLIC_KEY` | Yes      | Langfuse project public key          |
| `LANGFUSE_SECRET_KEY` | Yes      | Langfuse project secret key          |
| `LANGFUSE_HOST`       | No       | Langfuse host URL (defaults to cloud)|
| `INKEEP_API_KEY`      | Yes      | InKeep API key for retrieval calls   |

## Output formats

Results can be exported in any combination via the `formats` config field:

| Format     | Description                                                        |
|------------|--------------------------------------------------------------------|
| `langfuse` | Evaluation objects stored in the Langfuse experiment run (always on)|
| `yaml`     | Timestamped YAML file in `results_dir` with structured metrics     |

Output files are named `<YYYYMMDD_HHMMSS>_run_results.yaml` and written to the
directory specified by `results_dir` in the config.

Task-specific artifacts are also written alongside YAML:

| Artifact                                 | Task             | Contents                             |
|------------------------------------------|------------------|--------------------------------------|
| `<timestamp>_misclassifications.csv`     | `classification` | Misclassified items (input, predicted, expected) |
| `<timestamp>_bias_report.csv`            | `retrieval`      | Top over-retrieved URLs with bias scores |
