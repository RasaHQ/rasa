import glob
import os
import json
import pandas as pd
from pathlib import Path

REPORT_FOLDER = 'report'
TRAIN_TIME_FILENAME = 'training_metadata__times.csv'
INTENT_REPORT_FILENAME = 'intent_report.json'


def _format_hyperparameter(value : str) -> Union[str, int, float]:
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except:
        return value
    
def nested_dict_to_multiindex_df(row : Dict[str, Dict[str, Any]]):
    """Converts the given nested dict to a dataframe.
    Args:
        row: nested dict (with exactly two levels)
    """
    
def collect_experiment(experiment_dir : str):
    """Collect result for all runs in a nested dict (exactly two levels)."""
    script, data_name = Path(experiment_dir).name.split('__')      
    results = []
    for run_dir in glob.glob(os.path.join(experiment_dir, '*')):
        row = collect_run(run_dir)
        row['exp'] = {'script' : script, 'data' : data_name, 'timestamp' : timestamp }
        
        df_row = pd.DataFrame(pd.DataFrame(row).transpose().stack()).transpose()
        results.append(df_row)
    return pd.concat(results)

    
def collect_run(run_dir : str) -> Dict[str, Dict[str, Any]]:
    """Collects results for the given run in a nested dict (exactly two levels).
    
    Args:
        run_dir: 
    """
    row = dict()
    
    # run description
    params, timestamp = os.path.basename(run_dir).split('__')
    params_split = [param.split(':') for param in params.split(',')]
    params_split = [(param[0], _format_hyperparameter(param[1])) for param in params_split]
    row['param'] = {f'{param[0]}' : param[1] for param in params_split}
    
    # train times
    train_time_file = os.path.join(run, REPORT_FOLDER, TRAIN_TIME_FILENAME)
    if os.path.exists(train_time_file):
        times = pd.read_csv(train_time_file)
        for row_idx in range(len(times)):
            component = times["name"].iloc[row_idx]
            row[component] = {col_name : times[col_name].iloc[row_idx] for col_name in times.columns if col_name != "name"}
        
    # intent report
    def format_intent_report_key(key: str) -> str:
        if 'avg' in key:
            return key
        else:
            return f"intent_{key}"
    
    intent_report_file = os.path.join(run, report_folder, intent_report_file_name)
    if os.path.exists(intent_report_file):
        with open(intent_report_file, 'r') as f:
            intent_report  = json.load(f)
        row.update({key : value for key, value in intent_report.items() if isinstance(value, dict)})
        row.update({key : {'-' : value} for key, value in intent_report.items() if not isinstance(value, dict)})
        
    return row