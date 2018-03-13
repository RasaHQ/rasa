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
from typing import Optional

from rasa_nlu.components import ComponentBuilder
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.training_data import load_data

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_nlu.persistor import Persistor


def create_argparser():
    parser = argparse.ArgumentParser(
            description='train a custom language parser')

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
    parser.add_argument('--fixed_model_name',
                        help="If present, a model will always be persisted "
                             "in the specified directory instead of creating "
                             "a folder like 'model_20171020-160213'")
    parser.add_argument('-m', '--mitie_file', default=None,
                        help='File with mitie total_word_feature_extractor')
    return parser


class TrainingException(Exception):
    """Exception wrapping lower level exceptions that may happen while training

      Attributes:
          failed_target_project -- name of the failed project
          message -- explanation of why the request is invalid
      """

    def __init__(self, failed_target_project=None, exception=None):
        self.failed_target_project = failed_target_project
        if exception:
            self.message = exception.args[0]

    def __str__(self):
        return self.message


def create_persistor(config):
    # type: (RasaNLUConfig) -> Optional[Persistor]
    """Create a remote persistor to store the model if configured."""

    if config.get("storage") is not None:
        from rasa_nlu.persistor import get_persistor
        return get_persistor(config)
    else:
        return None


def init():  # pragma: no cover
    # type: () -> RasaNLUConfig
    """Combines passed arguments to create rasa NLU config."""

    parser = create_argparser()
    args = parser.parse_args()
    config = RasaNLUConfig(args.config, os.environ, vars(args))
    return config


def do_train_in_worker(config):
    # type: (RasaNLUConfig) -> Text
    """Loads the trainer and the data and runs the training in a worker."""

    try:
        _, _, persisted_path = do_train(config)
        return persisted_path
    except Exception as e:
        logger.exception("Failed to train project '{}'.".format(
                config.get("project")))
        raise TrainingException(config.get("project"), e)


def do_train(config,  # type: RasaNLUConfig
             component_builder=None  # type: Optional[ComponentBuilder]
             ):
    # type: (...) -> Tuple[Trainer, Interpreter, Text]
    """Loads the trainer and the data and runs the training of the model."""

    # Ensure we are training a model that we can save in the end
    # WARN: there is still a race condition if a model with the same name is
    # trained in another subprocess
    trainer = Trainer(config, component_builder)
    persistor = create_persistor(config)
    training_data = load_data(config['data'], config['language'])
    interpreter = trainer.train(training_data)
    persisted_path = trainer.persist(config['path'], persistor,
                                     config['project'],
                                     config['fixed_model_name'])
    return trainer, interpreter, persisted_path


if __name__ == '__main__':
    config = init()
    logging.basicConfig(level=config['log_level'])

    do_train(config)
    logger.info("Finished training")
