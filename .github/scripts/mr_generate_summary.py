# Collect the results of the various model test runs which are done as part of
# the model regression CI pipeline and dump them as a single file artifact.
# This artifact will the then be published at the end of the tests.
import json
import os
from pathlib import Path

REPORTS_PATH = Path(os.environ['REPORTS_DIR'])
print("REPORTS_PATH", list(REPORTS_PATH.glob("*")))

SUMMARY_FILE = os.environ["SUMMARY_FILE"]
print("SUMMARY_FILE", SUMMARY_FILE)

# CONFIG = os.environ["CONFIG"]
# DATASET = os.environ["DATASET_NAME"]
# TYPE = os.environ["TYPE"]
# IS_EXTERNAL = os.environ["IS_EXTERNAL"]
# CONFIG_REPOSITORY = "training-data"
# CONFIG_REPOSITORY_BRANCH = os.environ["DATASET_REPOSITORY_BRANCH"]
# DATASET_REPOSITORY_BRANCH = os.environ["DATASET_REPOSITORY_BRANCH"]
task_mapping = {
    "intent_report.json": "intent_classification",
    "CRFEntityExtractor_report.json": "entity_prediction",
    "DIETClassifier_report.json": "entity_prediction",
    "response_selection_report.json": "response_selection",
    "story_report.json": "story_prediction",
}


def generate_json(file, task, data):
    global IS_EXTERNAL
    global DATASET_REPOSITORY_BRANCH

    if not DATASET in data:
        data = {DATASET: {CONFIG: {}}, **data}
    elif not CONFIG in data[DATASET]:
        data[DATASET] = {CONFIG: {}, **data[DATASET]}

    if str(IS_EXTERNAL).lower() in ("yes", "true", "t", "1"):
        IS_EXTERNAL = True
        DATASET_REPOSITORY_BRANCH = os.environ["EXTERNAL_DATASET_REPOSITORY_BRANCH"]
    else:
        IS_EXTERNAL = False

    data[DATASET][CONFIG] = {
        "external_dataset_repository": IS_EXTERNAL,
        "dataset_repository_branch": DATASET_REPOSITORY_BRANCH,
        "config_repository": CONFIG_REPOSITORY,
        "config_repository_branch": CONFIG_REPOSITORY_BRANCH,
        "dataset_commit": os.environ["DATASET_COMMIT"],
        "accelerator_type": os.environ["ACCELERATOR_TYPE"],
        "test_run_time": os.environ["TEST_RUN_TIME"],
        "train_run_time": os.environ["TRAIN_RUN_TIME"],
        "total_run_time": os.environ["TOTAL_RUN_TIME"],
        "type": TYPE,
        **data[DATASET][CONFIG],
    }

    data[DATASET][CONFIG][task] = {**read_results(file)}

    return data


def read_results(file):
    with open(file) as json_file:
        data = json.load(json_file)

        keys = [
            "accuracy",
            "weighted avg",
            "macro avg",
            "micro avg",
            "conversation_accuracy",
        ]
        result = {key: data[key] for key in keys if key in data}

    return result


if __name__ == "__main__":
    data = {}
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE) as json_file:
            data = json.load(json_file)

    for dirpath, dirnames, files in os.walk(os.environ["RESULT_DIR"]):
        for f in files:
            if f not in task_mapping.keys():
                continue

            data = generate_json(os.path.join(dirpath, f), task_mapping[f], data)

    with open(SUMMARY_FILE, "w") as f:
        json.dump(data, f, sort_keys=True, indent=2)
