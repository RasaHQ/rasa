import os
from pathlib import Path
import sys
from unittest import mock

sys.path.append(".github/scripts")
from mr_publish_results import (  # noqa: E402
    prepare_ml_metric,
    prepare_ml_metrics,
    transform_to_seconds,
    generate_json,
    prepare_datadog_tags,
)

EXAMPLE_CONFIG = "Sparse + BERT + DIET(seq) + ResponseSelector(t2t)"
EXAMPLE_DATASET_NAME = "financial-demo"

ENV_VARS = {
    "BRANCH": "my-branch",
    "PR_ID": "10927",
    "PR_URL": "https://github.com/RasaHQ/rasa/pull/10856/",
    "GITHUB_EVENT_NAME": "pull_request",
    "GITHUB_RUN_ID": "1882718340",
    "GITHUB_SHA": "abc",
    "GITHUB_WORKFLOW": "CI - Model Regression",
    "IS_EXTERNAL": "false",
    "DATASET_REPOSITORY_BRANCH": "main",
    "CONFIG": EXAMPLE_CONFIG,
    "DATASET_NAME": EXAMPLE_DATASET_NAME,
    "CONFIG_REPOSITORY_BRANCH": "main",
    "DATASET_COMMIT": "52a3ad3eb5292d56542687e23b06703431f15ead",
    "ACCELERATOR_TYPE": "CPU",
    "TEST_RUN_TIME": "1m54s",
    "TRAIN_RUN_TIME": "4m4s",
    "TOTAL_RUN_TIME": "5m58s",
    "TYPE": "nlu",
    "INDEX_REPETITION": "0",
    "HOST_NAME": "github-runner-2223039222-22df222fcd-2cn7d",
}


@mock.patch.dict(os.environ, ENV_VARS, clear=True)
def test_generate_json():
    f = Path(__file__).parent / "test_data" / "intent_report.json"
    result = generate_json(f, task="intent_classification", data={})
    assert isinstance(result[EXAMPLE_DATASET_NAME][EXAMPLE_CONFIG], list)

    actual = result[EXAMPLE_DATASET_NAME][EXAMPLE_CONFIG][0]["intent_classification"]
    expected = {
        "accuracy": 1.0,
        "weighted avg": {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": 28,
        },
        "macro avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 28},
    }
    assert expected == actual


def test_transform_to_seconds():
    assert 87.0 == transform_to_seconds("1m27s")
    assert 87.3 == transform_to_seconds("1m27.3s")
    assert 27.0 == transform_to_seconds("27s")
    assert 3627.0 == transform_to_seconds("1h27s")
    assert 3687.0 == transform_to_seconds("1h1m27s")


def test_prepare_ml_model_perf_metrics():
    results = [
        {
            "macro avg": {
                "precision": 0.8,
                "recall": 0.8,
                "f1-score": 0.8,
                "support": 14,
            },
            "micro avg": {
                "precision": 1.0,
                "recall": 0.7857142857142857,
                "f1-score": 0.88,
                "support": 14,
            },
            "file_name": "DIETClassifier_report.json",
            "task": "Entity Prediction",
        },
        {
            "accuracy": 1.0,
            "weighted avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 28,
            },
            "macro avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 28,
            },
            "file_name": "intent_report.json",
            "task": "Intent Classification",
        },
    ]
    metrics_ml = prepare_ml_metrics(results)
    assert len(metrics_ml) == 17


def test_prepare_ml_model_perf_metrics_simple():
    result = {
        "accuracy": 1.0,
        "weighted avg": {"precision": 1, "recall": 1.0, "f1-score": 1, "support": 28},
        "task": "Intent Classification",
    }
    metrics_ml = prepare_ml_metric(result)
    assert len(metrics_ml) == 5

    for _, v in metrics_ml.items():
        assert isinstance(v, float)

    key, value = "Intent Classification.accuracy", 1.0
    assert key in metrics_ml and value == metrics_ml[key]

    key, value = "Intent Classification.weighted avg.f1-score", 1.0
    assert key in metrics_ml and value == metrics_ml[key]


@mock.patch.dict(os.environ, ENV_VARS, clear=True)
def test_prepare_datadog_tags():
    tags_list = prepare_datadog_tags()
    assert "dataset:financial-demo" in tags_list
