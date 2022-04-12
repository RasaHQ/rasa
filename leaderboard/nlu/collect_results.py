import glob
import os
import json
from pathlib import Path
from typing import Union, Dict, Any

from ruamel import yaml
import pandas as pd

REPORT_FOLDER = "report"
TRAIN_TIME_FILENAME = "training_metadata__times.csv"
INTENT_REPORT_FILENAME = "intent_report.json"
HYDRA_FOLDER = ".hydra"
CONFIG_YAML = "config.yaml"

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
    _, timestamp = os.path.basename(run_dir).split("__")
    row[TAG_EXPERIMENT] = {"timestamp": timestamp}

    for flag, collector in [
        (True, collect_config),
        (intents, collect_intent_report),
        (timings, collect_timings),
    ]:
        if flag:
            # NOTE: top keys are mutually distinct
            row.update(collector(run_dir))

    return row


def collect_config(run_dir: str) -> Dict[str, Dict[str, Any]]:
    row = {}
    config_yaml = os.path.join(run_dir, HYDRA_FOLDER, CONFIG_YAML)
    if os.path.exists(config_yaml):
        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)
        for key in ["model", "data"]:
            row[key] = config.pop(key)
        row[TAG_HYPERPARAM] = config
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
        row = collect_run(run_dir, timings=timings, intents=intents)
        row.setdefault("exp", {}).update({"script": script, "data": data_name})
        df_row = collection_to_df(row)
        results.append(df_row)
    df = pd.concat(results)
    add_total_train_times(df)
    df.sort_index(inplace=True, axis=1)
    df.reset_index(inplace=True, drop=True)
    return df


def add_total_train_times(df: pd.DataFrame) -> None:
    """Adds total training times for classifiers - and non-classifiers."""
    train_time_cols = [
        (top_level, low_level)
        for top_level, low_level in df.columns
        if top_level.startswith("train_") and low_level.strip() == "duration(sec)"
    ]
    top_level = "times"
    df[(top_level, "train_non_classifiers")] = sum(
        df[col].fillna(0) for col in train_time_cols if "Classifier" not in col[0]
    )
    df[(top_level, "train_all_classifiers")] = sum(
        df[col].fillna(0) for col in train_time_cols if "Classifier" in col[0]
    )
    df[(top_level, "train_all")] = sum(df[col].fillna(0) for col in train_time_cols)
