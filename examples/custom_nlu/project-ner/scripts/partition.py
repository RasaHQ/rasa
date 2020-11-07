import typer
from pathlib import Path
import bz2
import random
import shutil


def main(data_loc: Path, out_dir: Path, fraction_test: float = 0.2, fraction_dev: float = 0.1, seed: int = 0):
    """Partition the data into train/test/dev split."""
    random.seed(0)
    lines = (
        open(str(data_loc), mode="rt", encoding="utf8").read().strip().split("\n")
    )
    lines = [line for line in lines if line.strip()]
    random.shuffle(lines)
    dev_size = int(fraction_dev * len(lines))
    test_size = int(fraction_test * len(lines))
    train_size = (len(lines) - dev_size) - test_size
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)
    with (out_dir / "train.iob").open("w", encoding="utf8") as file_:
        file_.write("\n".join(lines[:train_size]))
    with (out_dir / "dev.iob").open("w", encoding="utf8") as file_:
        file_.write("\n".join(lines[train_size : train_size + dev_size]))
    with (out_dir / "test.iob").open("w", encoding="utf8") as file_:
        file_.write("\n".join(lines[train_size + dev_size :]))


if __name__ == "__main__":
    typer.run(main)
