"""
This script tests if the model architectures from `transformers` that we load in
`LanguageModelFeaturizer` work as expected. Particularly, if the tokenizer specific
cleanup of delimiter tokens (e.g. `##`) is successful.

In case of updates to `transformers` it should be run to ensure that there were no
changes to the underlying tokenizer implementation that would break our integration.
The model architectures that are not supported, as of `transformers` 4.13.0 are
listed in the `LanguageModelFeaturizer.INCOMPATIBLE_MODELS` and explicitly checked
when the component gets loaded.

Note that new versions of `transformers` might introduce entirely new architectures
that will not be captured by this test. In order to check which of these are
supported, and which should be added to the list of `INCOMPATIBLE_MODELS` and the
documentation, a manual run through
`transformers.tokenization_utils_base.PretrainedTokenizer.__subclasses__()` is
necessary. This will return the actual tokenizer classes, for which a model of
pretrained weights has to be identified and added to the list below.
"""

import sys
import logging
import uuid
from typing import Text

from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Those model weights cover all supported model architectures as of transformers 4.13.0
model_weights_list = [
    "rasa/LaBSE",
    "bert-base-uncased",
    "openai-gpt",
    "gpt2",
    "xlnet-base-cased",
    "xlm-mlm-enfr-1024",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "xlm-roberta-base",
    "microsoft/layoutlm-base-uncased",
    "cl-tohoku/bert-base-japanese",
    "YituTech/conv-bert-base",
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-reader-single-nq-base",
    "google/electra-small-generator",
    "funnel-transformer/small",
    "unc-nlp/lxmert-base-uncased",
    "google/mobilebert-uncased",
    "yjernite/retribert-base-uncased",
    "squeezebert/squeezebert-uncased",
    "facebook/bart-large-mnli",
    "facebook/blenderbot-400M-distill",
    "allenai/longformer-base-4096",
    "studio-ousia/luke-base",
    "microsoft/deberta-base",
    "moussaKam/mbarthez",
    "google/bigbird-roberta-base",
    "camembert-base",
    "TsinghuaAI/CPM-Generate",
    "microsoft/deberta-v2-xlarge",
    "flaubert/flaubert_small_cased",
    "allegro/herbert-base-cased",
    "google/fnet-base",
    "stas/tiny-wmt19-en-de",
    "facebook/mbart-large-50-one-to-many-mmt",
    "facebook/m2m100_418M",
    "sshleifer/tiny-marian-en-de",
    "studio-ousia/mluke-large",
    "microsoft/mpnet-base",
    "google/pegasus-xsum",
    "microsoft/prophetnet-large-uncased",
    "google/rembert",
    "junnyu/roformer_chinese_small",
    "tau/splinter-base",
]


def validate_tokenizer_prefixes_cleanup(model_weights: Text) -> bool:
    config = {"model_weights": model_weights, "load_model": False}
    featurizer = LanguageModelFeaturizer(
        {**LanguageModelFeaturizer.get_default_config(), **config},
        ExecutionContext(GraphSchema({}), uuid.uuid4().hex),
    )

    # a random long word that should get split by all tokenizers
    word = "incomprehensibilities"
    split_token_ids, split_token_strings = featurizer._lm_tokenize(word)
    token_ids, token_strings = featurizer._lm_specific_token_cleanup(
        split_token_ids, split_token_strings
    )

    # need `strip` here since `<\w>` is replaced by " " also at the end of the word
    return "".join(token_strings).strip() == word


if __name__ == "__main__":

    for model_weights in model_weights_list:
        logger.info(f"Checking model weights {model_weights}")
        if not validate_tokenizer_prefixes_cleanup(model_weights):
            logger.error(f"Tokenizer check failed for {model_weights}")
            sys.exit(1)

    logger.info("All tokenizers successful")
    sys.exit(0)
