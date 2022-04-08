import glob
import os
from pathlib import Path
import copy
from datetime import datetime
import click
import random
import logging
import sys

from omegaconf import MISSING

from leaderboard.utils import experiment
from leaderboard.nlu import exp_3_stratify_intents_only_for_exclusion as experiment_type

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

DRY_RUN = False

# =============================================================
# Directories
# =============================================================

rasa_dir = os.path.abspath("../../../../")
# rasa_dir = "/home/jupyter/rasa/"
train_data_dir = os.path.abspath("../../../../../training-data")
# train_data_dir = "/home/jupyter/training_data/"
root_out_dir = os.path.abspath("../../../../../results")
# root_out_dir = "/home/jupyter/results/"
config_dir = os.path.join(rasa_dir, "leaderboard", "configs", "intent_classification")

for folder in [rasa_dir, train_data_dir, root_out_dir, config_dir]:
    assert Path(folder).exists(), f"Expected a directory at {folder}"

run_script_name = Path(__file__).name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
out_dir = Path(root_out_dir) / f"{run_script_name}_{timestamp}"

if not DRY_RUN:
    out_dir.mkdir(parents=True, exist_ok=True)

logging.info(f"Rasa Repo: {rasa_dir}")
logging.info(f"Training Data Repo: {train_data_dir}")
logging.info(f"Root Output Directory: {root_out_dir}")
logging.info(f"Output Directory: {out_dir}")

# =============================================================
# Configurations
# =============================================================

data_dir = os.path.join(train_data_dir, "public", "HERMIT", "KFold_1")
data_name = "HERMIT-KFold-1"
train_data = os.path.join(data_dir, "train", "train.yml")
test_data = os.path.join(data_dir, "test", "test.yml")

for file in [train_data, test_data]:
    assert Path(file).is_file(), f"expected file {file}"

base_config = experiment_type.Config(
    train_exclusion_fraction=MISSING,
    data=experiment.DataConfig(
        name=f"{data_name}",
        data_path=train_data,
    ),
    model=experiment.ModelConfig(),
    clear_rasa_cache=True,
    test_data_path=test_data,
)

config_files = sorted(glob.glob(os.path.join(config_dir, "*")))


rng = random.Random(345)

variations = [
    [
        {("exclusion_seed",): value}
        for value in [rng.randint(1, 1000) for _ in range(10)]
    ],
    [{("train_exclusion_fraction",): value} for value in [0.0, 0.75, 0.25, 0.5, 0.25]],
    [
        {
            ("model", "name"): Path(config_file).name.replace(".yml", ""),
            ("model", "config_path"): config_file,
        }
        for config_file in config_files
    ],
]

configs = [base_config]
for idx, variation in enumerate(variations):
    logging.info(f"{idx}. Varying: {variation}")
    configs = experiment.create_variations(configs, variation)

for config in configs:
    config.validate_no_missing()
    config.validate()

logging.info(f"=> Total # of Configurations = {len(configs)}")

# =============================================================
# To Run or Not to Run!
# =============================================================

if click.confirm("Do you want to continue?", default=True):
    logging.info("Let's run!")

    if not DRY_RUN:
        experiment.multirun(
            experiment_module=experiment_type,
            configs=configs,
            out_dir=str(out_dir),
            capture=False,
        )
