from pathlib import Path
import secrets

from typing import Text

import rasa
from rasa.shared.core.domain import Domain
from rasa.shared.utils.io import write_yaml


def _new_model_path_in_same_dir(old_model_path: Text) -> Text:
    return str(Path(old_model_path).parent / (secrets.token_hex(8) + ".tar.gz"))


def test_models_not_retrained_if_no_new_data(
    trained_e2e_model: Text,
    moodbot_domain_path: Path,
    e2e_bot_config_file: Path,
    e2e_stories_path: Text,
    nlu_data_path: Text,
    trained_e2e_model_cache: Path,
):
    result = rasa.train(
        str(moodbot_domain_path),
        str(e2e_bot_config_file),
        [e2e_stories_path, nlu_data_path],
        output=_new_model_path_in_same_dir(trained_e2e_model),
        dry_run=True,
    )

    assert result.code == 0


def test_dry_run_model_will_not_be_retrained_if_only_new_responses(
    trained_e2e_model: Text,
    moodbot_domain_path: Path,
    e2e_bot_config_file: Path,
    e2e_stories_path: Text,
    nlu_data_path: Text,
    trained_e2e_model_cache: Path,
    tmp_path: Path,
):
    domain = Domain.load(moodbot_domain_path)
    domain_with_extra_response = """
    version: '3.1'
    responses:
      utter_greet:
      - text: "Hi from Rasa"
    """
    domain_with_extra_response = Domain.from_yaml(domain_with_extra_response)

    new_domain = domain.merge(domain_with_extra_response)
    new_domain_path = tmp_path / "domain.yml"
    write_yaml(new_domain.as_dict(), new_domain_path)

    result = rasa.train(
        str(new_domain_path),
        str(e2e_bot_config_file),
        [e2e_stories_path, nlu_data_path],
        output=str(tmp_path),
        dry_run=True,
    )

    assert result.code == 0
