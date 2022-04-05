from pathlib import Path
import time

from rasa.engine.storage.local_model_storage import LocalModelStorage


def extract_metadata(model_archive_path: Path, out_folder: Path) -> None:
    """Extract data from the model's metadata to a separate report file."""
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
