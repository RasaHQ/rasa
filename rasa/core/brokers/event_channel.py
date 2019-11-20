import warnings

from rasa.core.brokers.broker import EventBroker


# noinspection PyAbstractClass
class EventChannel(EventBroker):
    warnings.warn(
        "Deprecated, inherit from `EventBroker` instead of `EventChannel`. "
        "The `EventChannel` class will be removed.",
        DeprecationWarning,
        stacklevel=2,
    )
