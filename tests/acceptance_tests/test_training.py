from pathlib import Path
import secrets

from typing import Text

import pytest
import rasa


def _new_model_path_in_same_dir(old_model_path: Text) -> Text:
    return str(Path(old_model_path).parent / (secrets.token_hex(8) + ".tar.gz"))


@pytest.mark.acceptance
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
