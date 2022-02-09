import argparse
import time
from typing import List, NamedTuple, Optional, Text, Tuple

from transformers import AutoTokenizer, TFAutoModel

import rasa.shared.utils.io
from rasa.nlu.utils.hugging_face.registry import model_weights_defaults

COMP_NAME = "LanguageModelFeaturizer"
DEFAULT_MODEL_NAME = "bert"


class CompMetadata(NamedTuple):
    """Holds information about component."""
    model_name: Optional[Text] = None
    model_weights: Text = 0


def get_model_name_and_weights_from_config(
    config_path: str,
) -> List[CompMetadata]:
    config = rasa.shared.utils.io.read_config_file(config_path)
    print(config)
    steps = config.get("pipeline", [])

    # Look for LanguageModelFeaturizer steps
    steps = list(filter(lambda x: x["name"] == COMP_NAME, steps))

    name_weight_tuples = []
    for lmfeat_step in steps:
        if "model_name" not in lmfeat_step:
            model_name = DEFAULT_MODEL_NAME
            model_weights = model_weights_defaults[DEFAULT_MODEL_NAME]
        else:
            model_name = lmfeat_step["model_name"]
            model_weights = lmfeat_step.get("model_weights", model_weights_defaults[model_name])
        name_weight_tuples.append(CompMetadata(model_name, model_weights))

    return name_weight_tuples


def instantiate_to_download(model_weights: Text) -> None:
    """Instantiates Auto class instances, but only to download."""
    _ = AutoTokenizer.from_pretrained(model_weights)
    print("Done with AutoTokenizer, now doing TFAutoModel")
    _ = TFAutoModel.from_pretrained(model_weights)


def download(config_path: str):
    name_weight_tuples = get_model_name_and_weights_from_config(config_path)

    if not name_weight_tuples:
        print(f"No {COMP_NAME} model_weights used for this config: Skipping download")
        return

    for name_weight_tuple in name_weight_tuples:
        print(f"model_name: {name_weight_tuple.model_name}, "
              f"model_weights: {name_weight_tuple.model_weights}")
        start = time.time()

        instantiate_to_download(name_weight_tuple.model_weights)

        duration_in_sec = time.time() - start
        print(f"Instantiating Auto classes takes {duration_in_sec:.2f}seconds")


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
