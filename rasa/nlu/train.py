import logging
import typing
from typing import Optional, Text

from rasa.nlu import utils
from rasa.shared.nlu.training_data.loading import load_data
from rasa.utils import io as io_utils
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
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
        logger.warning(
            f"Could not retrieve training data from URL. Using empty "
            f"training data instead. Error details:\n{e}"
        )
        return TrainingData()


def create_persistor(persistor: Optional[Text]) -> Optional["Persistor"]:
    """Create a remote persistor to store the model if configured."""

    if persistor is not None:
        from rasa.nlu.persistor import get_persistor

        return get_persistor(persistor)
    else:
        return None
