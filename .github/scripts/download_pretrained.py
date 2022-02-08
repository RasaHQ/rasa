import argparse
import time
from typing import Text, Tuple
import yaml

from transformers import AutoTokenizer, TFAutoModel

import rasa.shared.utils.io


COMP_NAME = 'LanguageModelFeaturizer'
MODEL_WEIGHTS_DEFAULT = {
    "bert": "rasa/LaBSE",
    "gpt": "openai-gpt",
    "gpt2": "gpt2",
    "xlnet": "xlnet-base-cased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}
# TODO replace with https://github.com/RasaHQ/rasa/blob/main/rasa/nlu/utils/hugging_face/registry.py#L58


def get_model_stuff_from_config(config_path: str) -> Tuple[Text, Text]:
    with open(config_path) as json_file:
        data = yaml.safe_load(json_file)

    config = rasa.shared.utils.io.read_config_file(config_path)
    steps = config["pipeline"]

    # Look for LanguageModelFeaturizer
    steps = list(filter(lambda x: x['name'] == COMP_NAME, steps))

    if len(steps) == 0:
        print(f"No {COMP_NAME} found")
        return None, None
    elif len(steps) > 1:
        print(f"Too many ({len(steps)}) {COMP_NAME}s found")
        return None, None

    lmfeat_step = steps[0]

    if 'model_name' not in lmfeat_step:
        return "bert", "rasa/LaBSE"
    model_name = lmfeat_step['model_name']

    if 'model_weights' not in lmfeat_step:
        return model_name,
    model_weights = lmfeat_step.get('model_weights', MODEL_WEIGHTS_DEFAULT['model_name'])

    return model_name, model_weights


def download(dataset: str, config_path: str, type: str):
    start = time.time()

    model_name, model_weights = get_model_stuff_from_config(config_path)  # Example config: bert_diet_responset2t.yml

    _ = AutoTokenizer.from_pretrained(model_weights)
    # don't use this tokenizer instance more, this was just to download pretrained weights

    _ = TFAutoModel.from_pretrained(model_weights)

    seconds = time.time() - start
    print(f'Downloading takes {seconds:.2f}seconds')


def create_argument_parser():
    """TODO"""

    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default=None,
        help="TODO",
    )

    return parser

if __name__ == "__main__":
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    download(cmdline_args.dataset, cmdline_args.config, cmdline_args.type)

