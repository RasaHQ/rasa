import analytics
import datetime
import json
import os

analytics.write_key = os.environ['SEGMENT_TOKEN']

def send_to_segment(context):
    jobID = os.environ['GITHUB_RUN_ID']

    analytics.identify(jobID, {
        'name': 'model-regression-tests',
        'created_at': datetime.datetime.now()
    })

    analytics.track(jobID, 'results', {
        'dataset': os.environ['DATASET_NAME'],
        'workflow': os.environ['GITHUB_WORKFLOW'],
        'config': os.environ['CONFIG'],
        'runner_type': os.environ['RUNNER_TYPE'],
        'test_run_time': os.environ['TEST_RUN_TIME'],
        'train_run_time': os.environ['TRAIN_RUN_TIME'],
        'total_run_time': os.environ['TOTAL_RUN_TIME'],
        'event': os.environ['GITHUB_EVENT_NAME'],
        **context
    })

def read_results(file):
    result = {}
    with open(file) as json_file:
        data = json.load(json_file)
        
        if "accuracy" in data:
            result["accuracy"] = data["accuracy"]

        if "weighted avg" in data:
            result["weighted_avg"] = data["weighted avg"]

        if "macro avg" in data:
            result["macro_avg"] = data["macro avg"]

        if "micro avg" in data:
            result["micro_avg"] = data["micro avg"]

    return result

def push_results(file_name, path):
    result = read_results(os.path.join(path, file_name))
    result["file_name"] = f
    send_to_segment(result)


for dirpath, dirnames, files in os.walk(os.environ['RESULT_DIR']):
    for f in files:
        if f.endswith("intent_report.json"):
            push_results(f, os.path.join(dirpath, f))
        elif f.endswith("CRFEntityExtractor_report.json"):
            push_results(f, os.path.join(dirpath, f))
        elif f.endswith("DIETClassifier_report.json"):
            push_results(f, os.path.join(dirpath, f))
        elif f.endswith("esponse_selection_report.json"):
            push_results(f, os.path.join(dirpath, f))

analytics.flush()