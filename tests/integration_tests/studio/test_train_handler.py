import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import rasa.studio.train
import rasa.dialogue_understanding.generator.flow_retrieval
from pytest import MonkeyPatch
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name


@pytest.mark.timeout(120, func_only=True)
@pytest.mark.parametrize(
    "args",
    [
        argparse.Namespace(
            assistant_name="test",
            domain="data/train/domain.yml",
            data=[
                "data/train/data/nlu.yml",
                "data/train/data/rules.yml",
                "data/train/data/stories.yml",
            ],
            config="data/train/config.yml",
            out=None,
            dry_run=False,
            augmentation=50,
            fixed_model_name="test_result",
            persist_nlu_data=None,
            force=False,
            epoch_fraction=None,
            finetune=None,
            entities=[],
            intents=[],
            remote_storage=None,
        )
    ],
)
def test_handle_train(
    args: argparse.Namespace,
    monkeypatch: MonkeyPatch,
) -> None:
    handler_mock = MagicMock()
    return_mock = MagicMock()
    return_mock.return_value = handler_mock
    handler_mock.has_flows = lambda: False
    handler_mock.has_nlu = lambda: True
    handler_mock.flows = None
    handler_mock.nlu = """version: "3.1"
nlu:
- intent: greet
  examples: |
    - hey
    - hello
    - hi
    - hello there
    - good morning
    - good evening
    - hey there
    - let's go
    - hey dude
    - good afternoon
- intent: goodbye
  examples: |
    - cu
    - good by
    - cee you later
    - good night
    - bye
    - goodbye
    - have a nice day
    - see you around
    - bye bye
    - see you later"""
    handler_mock.domain = """version: "3.1"
intents:
  - greet
  - goodbye
entities:
  - first_name
  - last_name"""
    monkeypatch.setattr(rasa.studio.train, "StudioDataHandler", return_mock)

    with TempDirectoryPath(get_temp_dir_name()) as temp_path:
        args.out = temp_path
        rasa.studio.train.handle_train(args)
        path = Path(temp_path, "test_result.tar.gz")
        assert path.is_file()
        assert path.exists()


@pytest.mark.parametrize(
    "args",
    [
        argparse.Namespace(
            assistant_name="test",
            domain="data/train/train_flows/domain.yml",
            data=[
                "data/train/train_flows/flows.yml",
            ],
            config="data/train/train_flows/config.yml",
            out=None,
            dry_run=False,
            augmentation=50,
            fixed_model_name="test_result_flows",
            persist_nlu_data=None,
            force=False,
            epoch_fraction=None,
            finetune=None,
            entities=[],
            intents=[],
            remote_storage=None,
        )
    ],
)
def test_handle_train_with_flows(
    args: argparse.Namespace,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test train handler")
    handler_mock = MagicMock()
    return_mock = MagicMock()
    return_mock.return_value = handler_mock
    handler_mock.has_flows = lambda: True
    handler_mock.has_nlu = lambda: False
    handler_mock.nlu = None
    handler_mock.flows = """flows:
  check_balance:
    name: check your balance
    description: check the user's account balance.

    steps:
      - action: check_balance
      - action: utter_current_balance
  """
    handler_mock.domain = """version: "3.1"
slots:
  current_balance:
    type: float
    mappings:
      - type: custom
responses:
  utter_current_balance:
    - text: You still have {current_balance} in your account.
actions:
  - check_balance"""
    monkeypatch.setattr(rasa.studio.train, "StudioDataHandler", return_mock)
    monkeypatch.setattr(
        rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval,
        "populate",
        lambda selff, flows, domain: None,
    )

    with TempDirectoryPath(get_temp_dir_name()) as temp_path:
        args.out = temp_path
        rasa.studio.train.handle_train(args)
        path = Path(temp_path, "test_result_flows.tar.gz")
        assert path.is_file()
        assert path.exists()
