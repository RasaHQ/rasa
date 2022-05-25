from copy import deepcopy
import sys
import tempfile
from pathlib import Path

import pytest
from ruamel.yaml import YAML

sys.path.append(".github/scripts")
import download_pretrained  # noqa: E402

CONFIG_FPATH = Path(__file__).parent / "test_data" / "bert_diet_response2t.yml"


def test_download_pretrained_lmf_exists_no_params():
    lmf_specs = download_pretrained.get_model_name_and_weights_from_config(CONFIG_FPATH)
    assert lmf_specs[0].model_name == "bert"
    assert lmf_specs[0].model_weights == "rasa/LaBSE"


def test_download_pretrained_lmf_exists_with_model_name():
    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG_FPATH)

    steps = config.get("pipeline", [])
    step = list(filter(lambda x: x["name"] == download_pretrained.COMP_NAME, steps))[0]
    step["model_name"] = "roberta"
    step["cache_dir"] = "/this/dir"

    with tempfile.NamedTemporaryFile("w+") as fp:
        yaml.dump(config, fp)
        fp.seek(0)
        lmf_specs = download_pretrained.get_model_name_and_weights_from_config(fp.name)
    assert lmf_specs[0].model_name == "roberta"
    assert lmf_specs[0].model_weights == "roberta-base"
    assert lmf_specs[0].cache_dir == "/this/dir"


def test_download_pretrained_unknown_model_name():
    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG_FPATH)

    steps = config.get("pipeline", [])
    step = list(filter(lambda x: x["name"] == download_pretrained.COMP_NAME, steps))[0]
    step["model_name"] = "unknown"

    with tempfile.NamedTemporaryFile("w+") as fp:
        yaml.dump(config, fp)
        fp.seek(0)
        with pytest.raises(KeyError):
            download_pretrained.get_model_name_and_weights_from_config(fp.name)


def test_download_pretrained_multiple_model_names():
    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG_FPATH)

    steps = config.get("pipeline", [])
    step = list(filter(lambda x: x["name"] == download_pretrained.COMP_NAME, steps))[0]
    step_new = deepcopy(step)
    step_new["model_name"] = "roberta"
    steps.append(step_new)

    with tempfile.NamedTemporaryFile("w+") as fp:
        yaml.dump(config, fp)
        fp.seek(0)
        lmf_specs = download_pretrained.get_model_name_and_weights_from_config(fp.name)
    assert len(lmf_specs) == 2
    assert lmf_specs[1].model_name == "roberta"


def test_download_pretrained_with_model_name_and_nondefault_weight():
    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG_FPATH)

    steps = config.get("pipeline", [])
    step = list(filter(lambda x: x["name"] == download_pretrained.COMP_NAME, steps))[0]
    step["model_name"] = "bert"
    step["model_weights"] = "bert-base-uncased"

    with tempfile.NamedTemporaryFile("w+") as fp:
        yaml.dump(config, fp)
        fp.seek(0)
        lmf_specs = download_pretrained.get_model_name_and_weights_from_config(fp.name)
    assert lmf_specs[0].model_name == "bert"
    assert lmf_specs[0].model_weights == "bert-base-uncased"


def test_download_pretrained_lmf_doesnt_exists():
    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG_FPATH)

    steps = config.get("pipeline", [])
    step = list(filter(lambda x: x["name"] == download_pretrained.COMP_NAME, steps))[0]
    steps.remove(step)

    with tempfile.NamedTemporaryFile("w+") as fp:
        yaml.dump(config, fp)
        fp.seek(0)
        lmf_specs = download_pretrained.get_model_name_and_weights_from_config(fp.name)
    assert len(lmf_specs) == 0
