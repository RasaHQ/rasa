import warnings

from rasa.core.brokers.file import FileEventBroker


class FileProducer(FileEventBroker):
    warnings.warn(
        "Deprecated, the class `FileProducer` has been renamed to `FileEventBroker`. "
        "The `FileProducer` class will be removed.",
        DeprecationWarning,
        stacklevel=2,
    )
