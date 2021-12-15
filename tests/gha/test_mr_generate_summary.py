from pathlib import Path
import sys

REPO_DIR = Path(__file__).parents[2]
sys.path.append(str(REPO_DIR / ".github/scripts"))
from mr_generate_summary import combine_result  # noqa: E402


RESULT1 = {
    "financial-demo": {
        "BERT + DIET(bow) + ResponseSelector(bow)": {
            "Entity Prediction": {"macro avg": {"f1-score": 0.7333333333333333,}},
            "test_run_time": "47s",
        }
    }
}


def test_same_ds_different_config():
    result2 = {
        "financial-demo": {
            "Sparse + DIET(bow) + ResponseSelector(bow)": {
                "Entity Prediction": {"macro avg": {"f1-score": 0.88,}},
                "test_run_time": "47s",
            }
        }
    }
    expected_combined = {
        "financial-demo": {
            "BERT + DIET(bow) + ResponseSelector(bow)": {
                "Entity Prediction": {"macro avg": {"f1-score": 0.7333333333333333,}},
                "test_run_time": "47s",
            },
            "Sparse + DIET(bow) + ResponseSelector(bow)": {
                "Entity Prediction": {"macro avg": {"f1-score": 0.88,}},
                "test_run_time": "47s",
            },
        }
    }

    actual_combined = combine_result(RESULT1, result2)
    assert actual_combined == expected_combined

    actual_combined = combine_result(result2, RESULT1)
    assert actual_combined == expected_combined


def test_different_ds_same_config():
    result2 = {
        "Carbon Bot": {
            "Sparse + DIET(bow) + ResponseSelector(bow)": {
                "Entity Prediction": {"macro avg": {"f1-score": 0.88,}},
                "test_run_time": "47s",
            }
        }
    }
    expected_combined = {
        "financial-demo": {
            "BERT + DIET(bow) + ResponseSelector(bow)": {
                "Entity Prediction": {"macro avg": {"f1-score": 0.7333333333333333,}},
                "test_run_time": "47s",
            },
        },
        "Carbon Bot": {
            "Sparse + DIET(bow) + ResponseSelector(bow)": {
                "Entity Prediction": {"macro avg": {"f1-score": 0.88,}},
                "test_run_time": "47s",
            }
        },
    }

    actual_combined = combine_result(RESULT1, result2)
    assert actual_combined == expected_combined

    actual_combined = combine_result(result2, RESULT1)
    assert actual_combined == expected_combined


def test_start_empty():
    result2 = {}
    expected_combined = {
        "financial-demo": {
            "BERT + DIET(bow) + ResponseSelector(bow)": {
                "Entity Prediction": {"macro avg": {"f1-score": 0.7333333333333333,}},
                "test_run_time": "47s",
            },
        }
    }

    actual_combined = combine_result(RESULT1, result2)
    assert actual_combined == expected_combined

    actual_combined = combine_result(result2, RESULT1)
    assert actual_combined == expected_combined
