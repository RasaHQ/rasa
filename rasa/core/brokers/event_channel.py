import warnings

from rasa.core.brokers.broker import EventBroker


# noinspection PyAbstractClass
class EventChannel(EventBroker):
    warnings.warn(
        "The `EventChannel` class is deprecated, please inherit from "
        "`EventBroker` instead. `EventChannel` will be removed "
        "in future Rasa versions.",
        DeprecationWarning,
        stacklevel=2,
    )
