import glob
import os
import json
from pathlib import Path
from typing import Union, Dict, Any

import pandas as pd

REPORT_FOLDER = "report"
TRAIN_TIME_FILENAME = "training_metadata__times.csv"
INTENT_REPORT_FILENAME = "intent_report.json"

TAG_EXPERIMENT = "exp"
TAG_HYPERPARAM = "param"
TAG_INTENT = "intent"
# no tag for times - cause node names indicate e.g. train mode


def _format_hyperparameter(value: str) -> Union[str, int, float]:
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except:
        return value


def collection_to_df(row: Dict[str, Dict[str, Any]]):
    """Converts a nested dict with exactly two levels to a dataframe.
    Args:
        row: nested dict (with exactly two levels)
    Returns:
        dataframe with multi-index columns
    """
    return pd.DataFrame(pd.DataFrame(row).transpose().stack()).transpose()


def collect_run(
    run_dir: str, timings: bool = True, intents: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Collects results for the given run in a nested dict.

    Args:
        run_dir: directory for an experiment run created via a
            `base_nlu_experiment.multirun` (for custom runs
            leveraging hydra multirun from commandline which result
            in result folders with different pattern, use the
            `collect_{timings|intent_report|...}` functions instead)
        intents: set to True to collect intent reports
        timings: set to True to collect train times
    Returns:
        a nested dict with exactly two levels
    """
    row = {}

    params, timestamp = os.path.basename(run_dir).split("__")
    params_split = [param.split(":") for param in params.split(",")]
    params_split = [
        (param[0], _format_hyperparameter(param[1])) for param in params_split
    ]
    row[TAG_EXPERIMENT] = {"timestamp": timestamp}
    row[TAG_HYPERPARAM] = {f"{param[0]}": param[1] for param in params_split}

    for flag, collector in [
        (intents, collect_intent_report),
        (timings, collect_timings),
    ]:
        if flag:
            # NOTE: top keys are mutually distinct
            row.update(collector(run_dir))

    return row


def collect_timings(run_dir: str) -> Dict[str, Dict[str, Any]]:
    row = {}
    train_time_file = os.path.join(run_dir, REPORT_FOLDER, TRAIN_TIME_FILENAME)
    if os.path.exists(train_time_file):
        times = pd.read_csv(train_time_file)
        for row_idx in range(len(times)):
            component = times["name"].iloc[row_idx]
            row[component] = {
                col_name: times[col_name].iloc[row_idx]
                for col_name in times.columns
                if col_name != "name"
            }
    return row


def collect_intent_report(run_dir: str) -> Dict[str, Dict[str, Any]]:
    row = {}

    def format_intent_report_key(key: str) -> str:
        if "avg" in key:
            return key
        else:
            return f"{TAG_INTENT}_{key}"

    intent_report_file = os.path.join(run_dir, REPORT_FOLDER, INTENT_REPORT_FILENAME)
    if os.path.exists(intent_report_file):
        with open(intent_report_file, "r") as f:
            intent_report = json.load(f)

        row.update(
            {
                format_intent_report_key(key): value
                for key, value in intent_report.items()
                if isinstance(value, dict)
            }
        )
        row.update(
            {
                format_intent_report_key(key): {"-": value}
                for key, value in intent_report.items()
                if not isinstance(value, dict)
            }
        )

    return row


def results2df(
    experiment_dir: str, timings: bool = True, intents: bool = True
) -> pd.DataFrame:
    """Collect result for all runs in a nested dict (exactly two levels)."""
    script, data_name = Path(experiment_dir).name.split("__")
    results = []
    for run_dir in glob.glob(os.path.join(experiment_dir, "*")):
        row = collect_run(run_dir)
        row.setdefault("exp", {}).update({"script": script, "data": data_name})
        df_row = collection_to_df(row)
        results.append(df_row)
    df = pd.concat(results)
    add_total_train_times(df)
    return df


def add_total_train_times(df: pd.DataFrame) -> None:
    """Adds total training times for classifiers - and non-classifiers."""
    train_time_cols = [
        col
        for col in df.columns
        if col[0].startswith("train_") and col[1].strip() == "duration(sec)"
    ]
    df["train_non_classifiers"] = sum(
        df[col].fillna(0) for col in train_time_cols if "Classifier" not in col[0]
    )
    df["train_all_classifiers"] = sum(
        df[col].fillna(0) for col in train_time_cols if "Classifier" in col[0]
    )
    df["train_all"] = sum(df[col].fillna(0) for col in train_time_cols)
