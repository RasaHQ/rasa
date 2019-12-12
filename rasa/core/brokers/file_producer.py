import warnings

from rasa.core.brokers.file import FileEventBroker


class FileProducer(FileEventBroker):
    warnings.warn(
        "The `FileProducer` class is deprecated, please inherit from "
        "`FileEventBroker` instead. `FileProducer` will be removed in "
        "future Rasa versions.",
        DeprecationWarning,
        stacklevel=2,
    )
