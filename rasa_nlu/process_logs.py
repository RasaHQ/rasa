from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import io
import json

import logging
from builtins import str

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data
from rasa_nlu.model import Metadata, Interpreter


pretrained_extractors = {"ner_duckling", "ner_spacy"}


def create_argparser():
    parser = argparse.ArgumentParser(
            description='Process logs from Rasa NLU server. If a model dir is specified, ' +
                        'load that model and re-do the predictions. Sort by intent confidence, ' +
                        'and output the data in the rasa json format for training data'
    )
    parser.add_argument('-m', '--model_dir', default=None,
                        help='dir containing model (optional)')
    parser.add_argument('-l', '--log_file',
                        help='file or dir containing training data')
    parser.add_argument('-o', '--out_file',
                        help='file where to save the logs in rasa format')
    return parser


def process_logs(model_directory, log_file, out_file):
    logged_predictions = io.open(log_file, encoding="utf-8").readlines()

    if model_directory is not None:
        # load model & its training data
        interpreter = Interpreter.load(model_directory, RasaNLUConfig())

        logged_texts = set(logged_predictions)
        # dedupe & create test set
        # predict on test set
        predictions = []
        for t in logged_predictions:
            cleaned_text = t.strip(" \n")
            if not cleaned_text.startswith("_"):
                predictions.append(interpreter.parse(cleaned_text))
    else:
        predictions = []

    predictions = [p for p in predictions if p.get("intent", {}).get("confidence") is not None]
    predictions.sort(key=lambda p: p["intent"]["confidence"])

    preds = []
    for p in predictions:
        entities = []
        for e in p.get("entities", []):
            if e.get("extractor") not in pretrained_extractors:
                entities.append({k: e[k] for k in ('start', 'end', 'entity', 'value')})
        e = {
            "intent": p["intent"]["name"],
            "entities": entities,
            "text": p["text"]
        }
        if p["intent"]["name"] == "faq":
            e["refinement"] = p["faq"]["name"]
        preds.append(e)
    data = {"rasa_nlu_data": {"common_examples": preds}}

    # persist
    with io.open(out_file, "w", encoding="utf-8") as f:
        f.write(str(json.dumps(data, indent=2, ensure_ascii=False)))


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    parser = create_argparser()
    args = parser.parse_args()
    process_logs(args.model_dir, args.log_file, args.out_file)
