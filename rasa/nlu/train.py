import logging
import typing
from pathlib import Path
from typing import Any, Optional, Text, Tuple, Union, Dict

import rasa.shared.utils.common
from rasa.nlu import config, utils
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer
from rasa.shared.nlu.training_data.loading import load_data
from rasa.utils import io as io_utils
from rasa.utils.endpoints import EndpointConfig


if typing.TYPE_CHECKING:
    from rasa.shared.importers.importer import TrainingDataImporter
    from rasa.shared.nlu.training_data.training_data import NLUTrainingDataFull

logger = logging.getLogger(__name__)


class TrainingException(Exception):
    """Exception wrapping lower level exceptions that may happen while training

    Attributes:
        failed_target_project -- name of the failed project
        message -- explanation of why the request is invalid
    """

    def __init__(
        self,
        failed_target_project: Optional[Text] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        self.failed_target_project = failed_target_project
        if exception:
            self.message = exception.args[0]
        else:
            self.message = ""
        super(TrainingException, self).__init__()

    def __str__(self) -> Text:
        return self.message


async def load_data_from_endpoint(
    data_endpoint: EndpointConfig, language: Optional[Text] = "en"
) -> "NLUTrainingDataFull":
    """Load training data from a URL."""
    import requests

    if not utils.is_url(data_endpoint.url):
        raise requests.exceptions.InvalidURL(data_endpoint.url)
    try:
        response = await data_endpoint.request("get")
        response.raise_for_status()
        temp_data_file = io_utils.create_temporary_file(response.content, mode="w+b")
        training_data = load_data(temp_data_file, language)

        return training_data
    except Exception as e:
        logger.warning(f"Could not retrieve training data from URL:\n{e}")


def create_persistor(persistor: Optional[Text]):
    """Create a remote persistor to store the model if configured."""

    if persistor is not None:
        from rasa.nlu.persistor import get_persistor

        return get_persistor(persistor)
    else:
        return None


async def train(
    nlu_config: Union[Text, Dict, RasaNLUModelConfig],
    data: Union[Text, "TrainingDataImporter"],
    path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    storage: Optional[Text] = None,
    component_builder: Optional[ComponentBuilder] = None,
    training_data_endpoint: Optional[EndpointConfig] = None,
    persist_nlu_training_data: bool = False,
    **kwargs: Any,
) -> Tuple[Trainer, Interpreter, Optional[Text]]:
    """Loads the trainer and the data and runs the training of the model."""
    if not isinstance(nlu_config, RasaNLUModelConfig):
        nlu_config = config.load(nlu_config)

    # Ensure we are training a model that we can save in the end
    # WARN: there is still a race condition if a model with the same name is
    # trained in another subprocess
    trainer = Trainer(nlu_config, component_builder)
    persistor = create_persistor(storage)

    training_data = await _load_training_data(data, nlu_config, training_data_endpoint)

    interpreter = trainer.train(training_data, **kwargs)

    if path:
        persisted_path = trainer.persist(
            path, persistor, fixed_model_name, persist_nlu_training_data
        )
    else:
        persisted_path = None

    return trainer, interpreter, persisted_path


async def _load_training_data(
    data: Union[Text, "TrainingDataImporter"],
    model_config: RasaNLUModelConfig,
    training_data_endpoint: Optional[EndpointConfig] = None,
) -> "NLUTrainingDataFull":
    from rasa.shared.importers.importer import TrainingDataImporter

    if training_data_endpoint is not None:
        training_data = await load_data_from_endpoint(
            training_data_endpoint, model_config.language
        )
    elif isinstance(data, TrainingDataImporter):
        training_data = await data.get_nlu_data(model_config.language)
    else:
        training_data = load_data(data, model_config.language)

    training_data.print_stats()

    if training_data.entity_roles_groups_used():
        rasa.shared.utils.common.mark_as_experimental_feature(
            "Entity Roles and Groups feature"
        )

    return training_data


async def train_in_chunks(
    model_config: Union[Text, Dict, RasaNLUModelConfig],
    training_data_importer: "TrainingDataImporter",
    number_of_chunks: int,
    train_path: Optional[Path] = None,
    fixed_model_name: Optional[Text] = None,
) -> Tuple[Trainer, Interpreter, Optional[Text]]:
    """Loads the trainer and the data and runs the training of the model in chunks.

    Args:
        model_config: The model configuration.
        training_data_importer: The training data importer.
        number_of_chunks: The number of chunks to use.
        train_path: The training path.
        fixed_model_name: The fixed model name.

    Returns:
        The trainer, the trained interpreter, and the path to the persisted model.
    """
    if not isinstance(model_config, RasaNLUModelConfig):
        model_config = config.load(model_config)

    trainer = Trainer(model_config)

    training_data = await _load_training_data(training_data_importer, model_config)

    interpreter, persisted_path = trainer.train_in_chunks(
        training_data, train_path, number_of_chunks, fixed_model_name
    )

    return trainer, interpreter, persisted_path
