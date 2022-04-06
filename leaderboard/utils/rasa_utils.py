import time
from pathlib import Path
from typing import Optional

import pandas as pd

from leaderboard.utils.base_experiment import absolute_path
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData

TAG_INTENT = "intent"
TAG_ENTITY = "entity"


def load_nlu_data(
    data_path: str,
    domain_path: Optional[str],
) -> TrainingData:
    """Load nlu training data."""
    data_path = absolute_path(data_path)
    if domain_path is not None:
        domain_path = absolute_path(domain_path)
    test_data_importer = TrainingDataImporter.load_from_dict(
        training_data_paths=[str(data_path)], domain_path=domain_path
    )
    nlu_data = test_data_importer.get_nlu_data()
    return nlu_data


def extract_metadata(model_archive_path: Path, out_folder: Path) -> None:
    """Saves some of the model's metadata to a separate report file."""
    meta_data = LocalModelStorage.metadata_from_archive(model_archive_path)

    timings_file = out_folder / "training_metadata__times.csv"
    if timings_file.is_file():
        raise RuntimeError(f"Output file {timings_file} exists already")

    with open(timings_file, "w") as f:
        f.write("name,start_time,duration(min),duration(sec)\n")
        for node in meta_data.start_times:
            date_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(meta_data.start_times[node])
            )
            duration_min = round(meta_data.durations[node] / 60, 4)
            f.write(f"{node},{date_time},{duration_min},{meta_data.durations[node]}\n")


def extract_nlu_stats(
    train: TrainingData, test: Optional[TrainingData], report_path: Path
) -> None:
    """Saves intent and entity counts in a separate report file."""

    stats = {}
    for split, data in [("train", train), ("test", test)]:
        if test is None:
            continue
        stats[split] = dict()
        stats[split]["len"] = len(data.nlu_examples)
        for description, counts in [
            (TAG_ENTITY, data.number_of_examples_per_entity),
            (TAG_INTENT, data.number_of_examples_per_intent),
        ]:
            for key, count in counts.items():
                split_dict = stats[f"{TAG_INTENT}_{key}"].setdefault(split, {})
                split_dict[split] = count

    out_file = report_path / f"stats.csv"
    if out_file.is_file():
        raise ValueError(f"File {out_file} exists already.")
    df = pd.DataFrame(pd.DataFrame(stats).transpose().stack()).transpose()
    df.to_csv(out_file)
