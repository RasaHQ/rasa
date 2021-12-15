from pathlib import Path
import sys

REPO_DIR = Path(__file__).parents[2]
sys.path.append(str(REPO_DIR / '.github/scripts'))
from mr_publish_results import (
    prepare_ml_metric, prepare_ml_metrics, transform_to_seconds)


def test_transform_to_seconds():
    assert 87. == transform_to_seconds('1m27s')
    assert 87.3 == transform_to_seconds('1m27.3s')
    assert 27. == transform_to_seconds('27s')
    assert 3627. == transform_to_seconds('1h27s')
    assert 3687. == transform_to_seconds('1h1m27s')


def test_prepare_ml_model_perf_metrics():
    results = [{
        'macro avg': {
            'precision': 0.8,
            'recall': 0.8,
            'f1-score': 0.8,
            'support': 14
        },
        'micro avg': {
            'precision': 1.0,
            'recall': 0.7857142857142857,
            'f1-score': 0.88,
            'support': 14
        },
        'file_name': 'DIETClassifier_report.json',
        'task': 'Entity Prediction'
    }, {
        'accuracy': 1.0,
        'weighted avg': {
            'precision': 1.0,
            'recall': 1.0,
            'f1-score': 1.0,
            'support': 28
        },
        'macro avg': {
            'precision': 1.0,
            'recall': 1.0,
            'f1-score': 1.0,
            'support': 28
        },
        'file_name': 'intent_report.json',
        'task': 'Intent Classification'
    }]
    metrics_ml = prepare_ml_metrics(results)
    assert len(metrics_ml) == 17


def test_prepare_ml_model_perf_metric():
    result = {
        'accuracy': 1.0,
        'weighted avg': {
            'precision': 1,
            'recall': 1.0,
            'f1-score': 1,
            'support': 28
        },
        'task': 'Intent Classification'
    }
    metrics_ml = prepare_ml_metric(result)
    assert len(metrics_ml) == 5

    for _, v in metrics_ml.items():
        assert isinstance(v, float)

    key, value = 'Intent Classification.accuracy', 1.0
    assert key in metrics_ml and value == metrics_ml[key]

    key, value = 'Intent Classification.weighted avg.f1-score', 1.0
    assert key in metrics_ml and value == metrics_ml[key]
