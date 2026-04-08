# Evaluator

Offline evaluation framework for Copilot components. Experiments run against
[Langfuse](https://langfuse.com) datasets, metrics are tracked in Langfuse and
exported locally as YAML/TXT. The framework is config-driven — switching between
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
│  Returns       │        │  item-level:   │
│  TaskResult    │        │   called per   │
│                │        │   item         │
└────────────────┘        └────────────────┘
        │                          │
        ▼                          ▼
┌──────────────────────────────────────────┐
│              Langfuse                    │
│  Dataset run + Evaluation objects        │
│  + local export (YAML, TXT)              │
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
| `task`         | Task type (available tasks: `classification`)     |
| `dataset_name` | Langfuse dataset to evaluate against              |
| `results_dir`  | Local directory for exported results              |
| `formats`      | Output formats: `langfuse`, `yaml`, `txt`         |

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
item and returns a `TaskResult`:

```python
class BaseTask(ABC):
    @abstractmethod
    async def run_task(self, *, item: Any, **kwargs: Any) -> Optional[TaskResult]: ...
```

- `__init__` — set up expensive resources once (e.g. model clients)
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
   - Initialize resources in `__init__`
   - Implement `run_task(*, item, **kwargs) -> Optional[TaskResult]`

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
   formats: [langfuse, yaml, txt]
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

## Environment variables

| Variable              | Required | Description                          |
|-----------------------|----------|--------------------------------------|
| `OPENAI_API_KEY`      | Yes      | OpenAI API key for LLM calls         |
| `LANGFUSE_PUBLIC_KEY` | Yes      | Langfuse project public key          |
| `LANGFUSE_SECRET_KEY` | Yes      | Langfuse project secret key          |
| `LANGFUSE_HOST`       | No       | Langfuse host URL (defaults to cloud) |

## Output formats

Results can be exported in any combination via the `formats` config field:

| Format     | Description                                                        |
|------------|--------------------------------------------------------------------|
| `langfuse` | Evaluation objects stored in the Langfuse experiment run (always on)|
| `yaml`     | Timestamped YAML file in `results_dir` with structured metrics     |
| `txt`      | Timestamped text file in `results_dir` with human-readable summary |

Output files are named `<YYYYMMDD_HHMMSS>_run_results.<ext>` and written to the
directory specified by `results_dir` in the config.
