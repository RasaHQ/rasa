from pathlib import Path
from _pytest.monkeypatch import MonkeyPatch
from typing import Text

import pytest

from rasa.core import training
from rasa.core.agent import Agent
from rasa.shared.core.domain import Domain

import rasa.model_training
import rasa.shared.utils.io


def test_load_training_data_reader_not_found_throws(tmp_path: Path, domain: Domain):
    (tmp_path / "file").touch()

    with pytest.raises(Exception):
        training.load_data(str(tmp_path), domain)


def test_training_script_with_restart_stories(tmp_path: Path, domain_path: Text):
    model_file = rasa.model_training.train_core(
        domain_path,
        config="data/test_config/max_hist_config.yml",
        stories="data/test_yaml_stories/stories_restart.yml",
        output=str(tmp_path),
        additional_arguments={},
    )

    assert Path(model_file).is_file()


@pytest.mark.timeout(160, func_only=True)
async def test_random_seed(
    tmp_path: Path, monkeypatch: MonkeyPatch, domain_path: Text, stories_path: Text
):
    policies_config = {
        "assistant_id": "placeholder_default",
        "policies": [{"name": "TEDPolicy", "random_seed": 42}, {"name": "RulePolicy"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(policies_config, config_file)

    model_file_1 = rasa.model_training.train_core(
        domain_path,
        config=str(config_file),
        stories=stories_path,
        output=str(tmp_path),
        additional_arguments={},
    )

    model_file_2 = rasa.model_training.train_core(
        domain_path,
        config=str(config_file),
        stories=stories_path,
        output=str(tmp_path),
        additional_arguments={},
    )

    processor_1 = Agent.load(model_file_1).processor
    processor_2 = Agent.load(model_file_2).processor

    probs_1 = await processor_1.predict_next_for_sender_id("1")
    probs_2 = await processor_2.predict_next_for_sender_id("2")
    assert probs_1["confidence"] == probs_2["confidence"]
