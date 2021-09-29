import random
from rasa.shared.core.trackers import DialogueStateTracker
from typing import Any, Iterable, List, Text, Optional
from rasa.core.tracker_store import TrackerStore


class MarkerTrackerLoader:
    """Represents a wrapper around a `TrackerStore` that loads trackers in a configurable access pattern."""

    @staticmethod
    def strategy_all(count: int, keys: Iterable[Text]) -> Iterable[Text]:
        """Selects all keys from the set of keys."""
        return keys

    @staticmethod
    def strategy_first_n(count: int, keys: Iterable[Text]) -> Iterable[Text]:
        """Takes the first N keys from the set of keys."""
        return keys[:count]

    @staticmethod
    def strategy_sample(count: int, keys: Iterable[Text]) -> Iterable[Text]:
        """Takes a sample of N keys from the set of keys."""
        return random.choices(keys, k=count)

    _STRATEGY_MAP = {
        "all": strategy_all,
        "first_n": strategy_first_n,
        "sample": strategy_sample,
    }

    def __init__(
        self,
        tracker_store: TrackerStore,
        strategy: str,
        count: int = None,
        seed: Any = None,
    ) -> None:
        """Create a MarkerTrackerLoader.
        
        Args:
            tracker_store: The underlying tracker store to access.
            strategy: The strategy to use for selecting trackers, can be 'all', 'sample', or 'first_n'.
            count: Number of trackers to return, can only be None if strategy is 'all'.
            seed: Optional seed to set up random number generator, only useful if strategy is 'sample'.
        """
        self.tracker_store = tracker_store
        self.count = count

        if strategy not in MarkerTrackerLoader._STRATEGY_MAP:
            raise  # Not sure what best exception type to use here is, subclass of RasaException?

        self.strategy = MarkerTrackerLoader._STRATEGY_MAP[strategy]

        if not count:
            if strategy != "all":
                raise  # Exception
            else:
                pass  # Raise warning - redundant parameter

        if seed:
            if strategy == "sample":
                random.seed(seed)
            else:
                pass  # Warning - redundant parameter

    # Not super happy with this - is it sufficient to pull out to optionals or do we unwrap
    # If we unwrap, do we need to make up the numbers
    # Do we want a generator version that yields one at a time to reduce memory demands?
    def load(self) -> List[Optional[DialogueStateTracker]]:
        """Load trackers according to strategy."""
        keys = self.strategy(self.tracker_store.keys())
        return [self.tracker_store.retrieve(sender) for sender in keys]
