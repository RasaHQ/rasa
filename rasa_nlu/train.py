from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import logging
import os

import typing
from typing import Text
from typing import Tuple

from rasa_nlu.components import ComponentBuilder
from rasa_nlu.converters import load_data
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer

from rasa_nlu.config import RasaNLUConfig
from typing import Optional

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_nlu.persistor import Persistor


def create_argparser():
    parser = argparse.ArgumentParser(description='train a custom language parser')

    parser.add_argument('-p', '--pipeline', default=None,
                        help="Pipeline to use for the message processing.")
    parser.add_argument('-o', '--path', default=None,
                        help="Path where model files will be saved")
    parser.add_argument('-d', '--data', default=None,
                        help="File containing training data")
    parser.add_argument('-c', '--config', required=True,
                        help="Rasa NLU configuration file")
    parser.add_argument('-l', '--language', default=None, choices=['de', 'en'],
                        help="Model and data language")
    parser.add_argument('-t', '--num_threads', default=None, type=int,
                        help="Number of threads to use during model training")
    parser.add_argument('-m', '--mitie_file', default=None,
                        help='File with mitie total_word_feature_extractor')
    return parser


def create_persistor(config):
    # type: (RasaNLUConfig) -> Optional[Persistor]
    """Create a remote persistor to store the model if the configuration requests it."""

    persistor = None
    if "bucket_name" in config:
        from rasa_nlu.persistor import get_persistor
        persistor = get_persistor(config)

    return persistor


def init():  # pragma: no cover
    # type: () -> RasaNLUConfig
    """Combines passed arguments to create rasa NLU config."""

    parser = create_argparser()
    args = parser.parse_args()
    config = RasaNLUConfig(args.config, os.environ, vars(args))
    return config


def do_train_in_worker(config):
    # type: (RasaNLUConfig) -> Text
    """Loads the trainer and the data and runs the training of the specified model in a subprocess."""

    _, _, persisted_path = do_train(config)
    return persisted_path


def do_train(config, component_builder=None):
    # type: (RasaNLUConfig, Optional[ComponentBuilder]) -> Tuple[Trainer, Interpreter, Text]
    """Loads the trainer and the data and runs the training of the specified model."""

    # Ensure we are training a model that we can save in the end
    # WARN: there is still a race condition if a model with the same name is trained in another subprocess
    trainer = Trainer(config, component_builder)
    persistor = create_persistor(config)
    training_data = load_data(config['data'])
    interpreter = trainer.train(training_data)
    persisted_path = trainer.persist(config['path'], persistor, model_name=config['name'])

    return trainer, interpreter, persisted_path


if __name__ == '__main__':
    config = init()
    logging.basicConfig(level=config['log_level'])

    do_train(config)
    logger.info("Finished training")
