import argparse
import logging
from typing import Any, Optional, Text, Tuple, Union

from rasa_nlu import config, utils
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Interpreter, Trainer
from rasa_nlu.training_data import load_data
from rasa_nlu.training_data.loading import load_data_from_endpoint
from rasa_nlu.utils import EndpointConfig, read_endpoints

logger = logging.getLogger(__name__)


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description='train a custom language parser')

    parser.add_argument('-o', '--path',
                        default="models/nlu/",
                        help="Path where model files will be saved")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-d', '--data',
                       default=None,
                       help="Location of the training data. For JSON and "
                            "markdown data, this can either be a single file "
                            "or a directory containing multiple training "
                            "data files.")

    group.add_argument('-u', '--url',
                       default=None,
                       help="URL from which to retrieve training data.")

    group.add_argument('--endpoints',
                       default=None,
                       help="EndpointConfig defining the server from which "
                            "pull training data.")

    parser.add_argument('-c', '--config',
                        required=True,
                        help="Rasa NLU configuration file")

    parser.add_argument('-t', '--num_threads',
                        default=1,
                        type=int,
                        help="Number of threads to use during model training")

    parser.add_argument('--project',
                        default=None,
                        help="Project this model belongs to.")

    parser.add_argument('--fixed_model_name',
                        help="If present, a model will always be persisted "
                             "in the specified directory instead of creating "
                             "a folder like 'model_20171020-160213'")

    parser.add_argument('--storage',
                        help='Set the remote location where models are stored. '
                             'E.g. on AWS. If nothing is configured, the '
                             'server will only serve the models that are '
                             'on disk in the configured `path`.')

    utils.add_logging_option_arguments(parser)
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


def create_persistor(persistor: Optional[Text]):
    """Create a remote persistor to store the model if configured."""

    if persistor is not None:
        from rasa_nlu.persistor import get_persistor
        return get_persistor(persistor)
    else:
        return None


def do_train_in_worker(cfg: RasaNLUModelConfig,
                       data: Text,
                       path: Text,
                       project: Optional[Text] = None,
                       fixed_model_name: Optional[Text] = None,
                       storage: Text = None,
                       component_builder: Optional[ComponentBuilder] = None

                       ):
    """Loads the trainer and the data and runs the training in a worker."""

    try:
        _, _, persisted_path = train(cfg, data, path, project,
                                     fixed_model_name, storage,
                                     component_builder)
        return persisted_path
    except BaseException as e:
        logger.exception("Failed to train project '{}'.".format(project))
        raise TrainingException(project, e)


def train(nlu_config: Union[Text, RasaNLUModelConfig],
          data: Text,
          path: Optional[Text] = None,
          project: Optional[Text] = None,
          fixed_model_name: Optional[Text] = None,
          storage: Optional[Text] = None,
          component_builder: Optional[ComponentBuilder] = None,
          training_data_endpoint: Optional[EndpointConfig] = None,
          **kwargs: Any
          ) -> Tuple[Trainer, Interpreter, Text]:
    """Loads the trainer and the data and runs the training of the model."""

    if isinstance(nlu_config, str):
        nlu_config = config.load(nlu_config)

    # Ensure we are training a model that we can save in the end
    # WARN: there is still a race condition if a model with the same name is
    # trained in another subprocess
    trainer = Trainer(nlu_config, component_builder)
    persistor = create_persistor(storage)
    if training_data_endpoint is not None:
        training_data = load_data_from_endpoint(training_data_endpoint,
                                                nlu_config.language)
    else:
        training_data = load_data(data, nlu_config.language)
    interpreter = trainer.train(training_data, **kwargs)

    if path:
        persisted_path = trainer.persist(path,
                                         persistor,
                                         project,
                                         fixed_model_name)
    else:
        persisted_path = None

    return trainer, interpreter, persisted_path


if __name__ == '__main__':
    cmdline_args = create_argument_parser().parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    if cmdline_args.url:
        data_endpoint = EndpointConfig(cmdline_args.url)
    else:
        data_endpoint = read_endpoints(cmdline_args.endpoints).data

    train(cmdline_args.config,
          cmdline_args.data,
          cmdline_args.path,
          cmdline_args.project,
          cmdline_args.fixed_model_name,
          cmdline_args.storage,
          training_data_endpoint=data_endpoint,
          num_threads=cmdline_args.num_threads)
    logger.info("Finished training")
