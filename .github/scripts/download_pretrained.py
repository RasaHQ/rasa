import argparse
import logging
import time
from typing import List, NamedTuple, Optional, Text

from transformers import AutoTokenizer, TFAutoModel

import rasa.nlu.featurizers.dense_featurizer.lm_featurizer
import rasa.shared.utils.io

logger = logging.getLogger(__name__)

COMP_NAME = "LanguageModelFeaturizer"


class LmfSpec(NamedTuple):
    """Holds information about the LanguageModelFeaturizer."""

    model_weights: Text
    cache_dir: Optional[Text] = None


def get_model_weights_from_config(
    config_path: str,
) -> List[LmfSpec]:
    config = rasa.shared.utils.io.read_config_file(config_path)
    logger.info(config)
    steps = config.get("pipeline", [])

    # Look for LanguageModelFeaturizer steps
    steps = list(filter(lambda x: x["name"] == COMP_NAME, steps))

    lmf_specs = []
    for lmfeat_step in steps:
        model_weights = (
            rasa.nlu.featurizers.dense_featurizer.lm_featurizer.get_model_weights(
                lmfeat_step
            )
        )
        cache_dir = lmfeat_step.get("cache_dir", None)
        lmf_specs.append(LmfSpec(model_weights, cache_dir))

    return lmf_specs


def instantiate_to_download(comp: LmfSpec) -> None:
    """Instantiates Auto class instances, but only to download."""

    _ = AutoTokenizer.from_pretrained(comp.model_weights, cache_dir=comp.cache_dir)
    logger.info("Done with AutoTokenizer, now doing TFAutoModel")
    _ = TFAutoModel.from_pretrained(comp.model_weights, cache_dir=comp.cache_dir)


def download(config_path: str):
    lmf_specs = get_model_weights_from_config(config_path)

    if not lmf_specs:
        logger.info(f"No {COMP_NAME} found, therefore, skipping download")
        return

    for lmf_spec in lmf_specs:
        logger.info(
            f"model_weights: {lmf_spec.model_weights}, "
            f"cache_dir: {lmf_spec.cache_dir}"
        )
        start = time.time()

        instantiate_to_download(lmf_spec)

        duration_in_sec = time.time() - start
        logger.info(f"Instantiating Auto classes takes {duration_in_sec:.2f}seconds")


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
