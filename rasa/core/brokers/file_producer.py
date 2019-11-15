import warnings

from rasa.core.brokers.file import FileEventBroker
from rasa.utils.common import raise_warning


class FileProducer(FileEventBroker):
    raise_warning(
        "The `FileProducer` class is deprecated, please inherit from "
        "`FileEventBroker` instead. `FileProducer` will be removed in "
        "future Rasa versions.",
        FutureWarning,
        docs="/api/event-brokers/",
    )
