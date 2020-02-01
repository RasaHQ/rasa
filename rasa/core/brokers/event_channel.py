from rasa.core.brokers.broker import EventBroker


# noinspection PyAbstractClass
from rasa.utils.common import raise_warning


class EventChannel(EventBroker):
    raise_warning(
        "The `EventChannel` class is deprecated, please inherit from "
        "`EventBroker` instead. `EventChannel` will be removed "
        "in future Rasa versions.",
        DeprecationWarning,
        stacklevel=2,
    )
