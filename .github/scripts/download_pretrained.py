import argparse
import time
from typing import Optional, Text, Tuple

from transformers import AutoTokenizer, TFAutoModel

import rasa.shared.utils.io
from rasa.nlu.utils.hugging_face.registry import model_weights_defaults

COMP_NAME = "LanguageModelFeaturizer"
DEFAULT_MODEL_NAME = "bert"


def get_model_name_and_weights_from_config(
    config_path: str,
) -> Tuple[Optional[Text], Optional[Text]]:
    config = rasa.shared.utils.io.read_config_file(config_path)
    print(config)
    steps = config.get("pipeline", [])

    # Look for LanguageModelFeaturizer
    steps = list(filter(lambda x: x["name"] == COMP_NAME, steps))

    if len(steps) == 0:
        print(f"No {COMP_NAME} found")
        return None, None
    elif len(steps) > 1:
        print(f"Too many ({len(steps)}) {COMP_NAME}s found")
        return None, None

    lmfeat_step = steps[0]

    if "model_name" not in lmfeat_step:
        return DEFAULT_MODEL_NAME, model_weights_defaults[DEFAULT_MODEL_NAME]
    model_name = lmfeat_step["model_name"]

    model_weights = lmfeat_step.get("model_weights", model_weights_defaults[model_name])

    return model_name, model_weights


def instantiate_to_download(model_weights: Text) -> None:
    """Instantiates Auto class instances, but only to download."""
    _ = AutoTokenizer.from_pretrained(model_weights)
    print("Done with AutoTokenizer, now doing TFAutoModel")
    _ = TFAutoModel.from_pretrained(model_weights)


def download(config_path: str):
    model_name, model_weights = get_model_name_and_weights_from_config(config_path)
    print(f"model_name: {model_name}, model_weights: {model_weights}")

    if not model_weights:
        print(f"No {COMP_NAME} model_weights used for this config: Skipping download")
        return

    start = time.time()
    instantiate_to_download(model_weights)

    seconds = time.time() - start
    print(f"Instantiating Auto classes takes {seconds:.2f}seconds")


def create_argument_parser() -> argparse.ArgumentParser:
    """Downloads pretrained models, i.e., Huggingface weights."""
    parser = argparse.ArgumentParser(
        description="Downloads pretrained models, i.e., Huggingface weights, "
        "e.g. path to bert_diet_responset2t.yml"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="The path to the config yaml file.",
    )
    return parser


if __name__ == "__main__":
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    download(cmdline_args.config)
