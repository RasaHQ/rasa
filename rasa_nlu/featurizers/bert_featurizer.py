from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import typing
from typing import Any
import os

from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from extract_features import main, create_features, model_fn_builder
from rasa_nlu import config

import tensorflow as tf
import modeling
import tokenization

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

tf.logging.set_verbosity(tf.logging.INFO)

def ndim(spacy_nlp):
    """Number of features used to represent a document / sentence."""
    # type: Language -> int
    return spacy_nlp.vocab.vectors_length


def features_for_doc(doc):
    """Feature vector for a single document / sentence."""
    # type: Doc -> np.ndarray
    return doc.vector


class BertFeaturizer(Featurizer):
    name = "intent_featurizer_bert"

    provides = ["text_features"]

    requires = []

    def __init__(self, component_config=None):
        if not component_config:
            component_config = {}

        # makes sure the name of the configuration is part of the config
        # this is important for e.g. persistence
        component_config["name"] = self.name
        self.component_config = config.override_defaults(
                self.defaults, component_config)

        self.partial_processing_pipeline = None
        self.partial_processing_context = None
        self.layer_indexes = [-2]

        model_dir = component_config.get("model_dir")
        print("Loading model from", model_dir)

        dir_files = os.listdir(model_dir)

        if all(file not in dir_files for file in ('bert_config.json', 'vocab.txt')):
            raise Exception("To use BertFeaturizer you need to specify a "
                            "directory path to a pre-trained model, i.e. "
                            "containing the files 'bert_config.json', "
                            "'vocab.txt' and model checkpoint")

        bert_config = modeling.BertConfig.from_json_file(os.path.join(model_dir, "bert_config.json"))
        self.tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            master=None,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=8,
                per_host_input_for_training=is_per_host))
        model_fn = model_fn_builder(
          bert_config=bert_config,
          init_checkpoint=os.path.join(model_dir, "bert_model.ckpt"),
          layer_indexes=self.layer_indexes,
          use_tpu=False,
          use_one_hot_embeddings=False)

        self.estimator = tf.contrib.tpu.TPUEstimator(
           use_tpu=False,
           model_fn=model_fn,
           config=run_config,
           predict_batch_size=8)

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None
        messages = [example.text for example in training_data.intent_examples]
        fs = create_features(messages, self.estimator, self.tokenizer, self.layer_indexes)
        features = []
        for x in fs:
            # features.append(np.array(x['features'][0]['layers'][0]['values']))
            feats = [y['layers'][0]['values'] for y in x['features'][1:-1]]
            features.append(np.average(feats, axis=0))
        for i, message in enumerate(training_data.intent_examples):
            message.set("text_features", features[i])
            # self._set_bert_features(example)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        self._set_bert_features(message)

    def _set_bert_features(self, message):
        """Adds the spacy word vectors to the messages text features."""
        # print(message)
        fs = create_features([message.text], self.estimator, self.tokenizer, self.layer_indexes)
        feats = [x['layers'][0]['values'] for x in fs[0]['features'][1:-1]]
        features = np.average(feats, axis=0)
        # features = np.array(fs[0]['features'][0]['layers'][0]['values'])
        message.set("text_features", features)
