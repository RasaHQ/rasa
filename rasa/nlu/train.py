import logging
import typing
from typing import Any, Optional, Text, Tuple, Union, Dict

from rasa.nlu import config, utils
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer
from rasa.shared.nlu.training_data.loading import load_data
from rasa.utils import io as io_utils
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from rasa.shared.importers.importer import TrainingDataImporter
    from rasa.shared.nlu.training_data.training_data import TrainingData
    from rasa.nlu.persistor import Persistor

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
) -> "TrainingData":
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


def create_persistor(persistor: Optional[Text]) -> Optional["Persistor"]:
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
    model_to_finetune: Optional[Interpreter] = None,
    **kwargs: Any,
) -> Tuple[Trainer, Interpreter, Optional[Text]]:
    """Loads the trainer and the data and runs the training of the model."""
    from rasa.shared.importers.importer import TrainingDataImporter

    if not isinstance(nlu_config, RasaNLUModelConfig):
        nlu_config = config.load(nlu_config)

    # Ensure we are training a model that we can save in the end
    # WARN: there is still a race condition if a model with the same name is
    # trained in another subprocess
    trainer = Trainer(
        nlu_config, component_builder, model_to_finetune=model_to_finetune
    )
    persistor = create_persistor(storage)
    if training_data_endpoint is not None:
        training_data = await load_data_from_endpoint(
            training_data_endpoint, nlu_config.language
        )
    elif isinstance(data, TrainingDataImporter):
        training_data = await data.get_nlu_data(nlu_config.language)
    else:
        training_data = load_data(data, nlu_config.language)

    training_data.print_stats()

    interpreter = trainer.train(training_data, **kwargs)

    if path:
        persisted_path = trainer.persist(
            path, persistor, fixed_model_name, persist_nlu_training_data
        )
    else:
        persisted_path = None

    return trainer, interpreter, persisted_path
