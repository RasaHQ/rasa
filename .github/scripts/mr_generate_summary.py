# Collect the results of the various model test runs which are done as part of
# the model regression CI pipeline and dump them as a single file artifact.
# This artifact will the then be published at the end of the tests.
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Dict, List


def combine_result(
    result1: Dict[str, dict], result2: Dict[str, Dict[str, Dict]]
) -> Dict[str, Dict[str, List]]:
    """Combines 2 result dicts to accumulated dict of the same format.

    Args:
        result1: dict of key: dataset, value: (dict of key: config, value: list of res)
                 Example: {
                              "Carbon Bot": {
                                  "Sparse + DIET(bow) + ResponseSelector(bow)": [{
                                      "Entity Prediction": {
                                          "macro avg": {
                                              "f1-score": 0.88,
                                          }
                                      },
                                      "test_run_time": "47s",
                                  }]
                              }
                          }
        result2: dict of key: dataset, value: (dict of key: config, value: list of res)

    Returns:
        dict of key: dataset, and value: (dict of key: config value: list of results)
    """
    combined_dict = defaultdict(lambda: defaultdict(list))
    for new_dict in [result1, result2]:
        for dataset, results_for_dataset in new_dict.items():
            for config, res in results_for_dataset.items():
                for res_dict in res:
                    combined_dict[dataset][config].append(res_dict)
    return combined_dict


if __name__ == "__main__":
    data = {}
    reports_dir = Path(os.environ["REPORTS_DIR"])
    reports_paths = list(reports_dir.glob("*/report.json"))

    for report_path in reports_paths:
        report_dict = json.load(open(report_path))
        data = combine_result(data, report_dict)

    summary_file = os.environ["SUMMARY_FILE"]
    with open(summary_file, "w") as f:
        json.dump(data, f, sort_keys=True, indent=2)
