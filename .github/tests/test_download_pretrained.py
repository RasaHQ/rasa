import sys
import tempfile
from pathlib import Path

from ruamel.yaml import YAML

sys.path.append(".github/scripts")
import download_pretrained  # noqa: E402

CONFIG_FPATH = Path(__file__).parent / "test_data" / "bert_diet_response2t.yml"


def test_download_pretrained_lmf_exists_without_params():
    name, weights = download_pretrained.get_model_name_and_weights_from_config(
        CONFIG_FPATH
    )
    assert name == "bert"
    assert weights == "rasa/LaBSE"


def test_download_pretrained_lmf_exists_with_model_name():
    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG_FPATH)

    steps = config.get("pipeline", [])
    step = list(filter(lambda x: x["name"] == download_pretrained.COMP_NAME, steps))[0]
    step["model_name"] = "roberta"

    with tempfile.NamedTemporaryFile("w+") as fp:
        yaml.dump(config, fp)
        fp.seek(0)
        name, weights = download_pretrained.get_model_name_and_weights_from_config(
            fp.name
        )
    assert name == "roberta"
    assert weights == "roberta-base"


def test_download_pretrained_lmf_exists_with_model_weight():
    name, weights = download_pretrained.get_model_name_and_weights_from_config(
        CONFIG_FPATH
    )

    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG_FPATH)

    steps = config.get("pipeline", [])
    step = list(filter(lambda x: x["name"] == download_pretrained.COMP_NAME, steps))[0]
    step["model_name"] = "roberta"
    step["model_weights"] = "abc"

    with tempfile.NamedTemporaryFile("w+") as fp:
        yaml.dump(config, fp)
        fp.seek(0)
        name, weights = download_pretrained.get_model_name_and_weights_from_config(
            fp.name
        )
    assert name == "roberta"
    assert weights == "abc"


def test_download_pretrained_lmf_doesnt_exists():
    name, weights = download_pretrained.get_model_name_and_weights_from_config(
        CONFIG_FPATH
    )

    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG_FPATH)

    steps = config.get("pipeline", [])
    step = list(filter(lambda x: x["name"] == download_pretrained.COMP_NAME, steps))[0]
    steps.remove(step)

    with tempfile.NamedTemporaryFile("w+") as fp:
        yaml.dump(config, fp)
        fp.seek(0)
        name, weights = download_pretrained.get_model_name_and_weights_from_config(
            fp.name
        )

    assert name is None
    assert weights is None
