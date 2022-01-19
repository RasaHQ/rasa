# Collect the results of the various model test runs which are done as part of
# the model regression CI pipeline and dump them as a single file artifact.
# This artifact will the then be published at the end of the tests.
import copy
import json
import os
from pathlib import Path
from typing import Dict


def combine_result(
    result1: Dict[str, dict], result2: Dict[str, dict]
) -> Dict[str, dict]:
    combined_dict = copy.deepcopy(result1)

    for dataset, results_for_dataset in result2.items():
        for config, res in results_for_dataset.items():

            if dataset not in combined_dict:
                combined_dict[dataset] = {}

            assert config not in combined_dict[dataset]
            combined_dict[dataset][config] = res
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
