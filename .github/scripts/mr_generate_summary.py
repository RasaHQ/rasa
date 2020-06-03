# Generate a file with a summary of test, the file will publish as an artifact
import json
import os

summary_file = os.environ['SUMMARY_FILE']
config = os.environ['CONFIG']
dataset = os.environ['DATASET']
task_mapping = {
    "intent_report.json": "Intent Classification",
    "CRFEntityExtractor_report.json": "Entity Prediction",
    "DIETClassifier_report.json": "Entity Prediction",
    "response_selection_report.json": "Response Selection"
}
data = {}

def generate_json(file, task):
    global data

    if (not dataset in data):
        data = { dataset: { config: {} }, **data}
    elif (not config in data[dataset]):
        data[dataset] = {
            config: {},
            **data[dataset]
        }

    data[dataset][config] = {
        'accelerator_type': os.environ['ACCELERATOR_TYPE'],
        'test_run_time': os.environ['TEST_RUN_TIME'],
        'train_run_time': os.environ['TRAIN_RUN_TIME'],
        'total_run_time': os.environ['TOTAL_RUN_TIME'],
        **data[dataset][config]
    }

    data[dataset][config][task] = {
        **read_results(file)
    }



def read_results(file):
    with open(file) as json_file:
        data = json.load(json_file)

        keys = ["accuracy", "weighted avg", "macro avg", "micro avg"]
        result = {key: data[key] for key in keys if key in data}

    return result

if __name__ == "__main__":
    if os.path.exists(summary_file):
        with open(summary_file) as json_file:
            data = json.load(json_file)

    for dirpath, dirnames, files in os.walk(os.environ['RESULT_DIR']):
        for f in files:
            if f.endswith("intent_report.json"):
                generate_json(os.path.join(dirpath, f), task_mapping[f])
            elif f.endswith("CRFEntityExtractor_report.json"):
                generate_json(os.path.join(dirpath, f), task_mapping[f])
            elif f.endswith("DIETClassifier_report.json"):
                generate_json(os.path.join(dirpath, f), task_mapping[f])
            elif f.endswith("response_selection_report.json"):
                generate_json(os.path.join(dirpath, f), task_mapping[f])

    with open(summary_file, 'w') as f:
        json.dump(data, f)
