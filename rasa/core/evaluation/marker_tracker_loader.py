import random
from rasa.shared.exceptions import RasaException
from rasa.shared.core.trackers import DialogueStateTracker
from typing import Any, Iterable, Iterator, List, Text, Optional
from rasa.core.tracker_store import TrackerStore
import rasa.shared.utils.io


def strategy_all(keys: List[Text], count: int) -> Iterable[Text]:
    """Selects all keys from the set of keys."""
    return keys


def strategy_first_n(keys: List[Text], count: int) -> Iterable[Text]:
    """Takes the first N keys from the set of keys."""
    return keys[:count]


def strategy_sample(keys: List[Text], count: int) -> Iterable[Text]:
    """Takes a sample of N keys from the set of keys."""
    return random.sample(keys, k=count)


class MarkerTrackerLoader:
    """Represents a wrapper over a `TrackerStore` with a configurable access pattern."""

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
        """Creates a MarkerTrackerLoader.

        Args:
            tracker_store: The underlying tracker store to access.
            strategy: The strategy to use for selecting trackers,
                      can be 'all', 'sample', or 'first_n'.
            count: Number of trackers to return, can only be None if strategy is 'all'.
            seed: Optional seed to set up random number generator,
                  only useful if strategy is 'sample'.
        """
        self.tracker_store = tracker_store

        if strategy not in MarkerTrackerLoader._STRATEGY_MAP:
            raise RasaException(
                "Invalid strategy for loading markers - '{strategy}' was given, \
                options 'all', 'sample', or 'first_n' exist"
            )

        self.strategy = MarkerTrackerLoader._STRATEGY_MAP[strategy]

        if strategy != "all":
            if not count:
                raise RasaException(
                    "Desired tracker count must be given for strategy '{strategy}'"
                )

            if count < 0:
                # Prefer exception here because if count is ever passed less than 0
                # something has definitely gone wrong, but if count exceeds population
                # size that might be someone overestimating how much data they have so
                # prefer to warn for that case and round down
                raise RasaException(
                    "Parameter 'count' must be greater than or equal to 0"
                )

        self.count = count

        if count and strategy == "all":
            rasa.shared.utils.io.raise_warning(
                "Parameter 'count' is ignored by strategy 'all'"
            )
            self.count = None

        if seed:
            if strategy == "sample":
                random.seed(seed)
            else:
                rasa.shared.utils.io.raise_warning(
                    "Parameter 'seed' is ignored by strategy '{strategy}'"
                )

    def load(self) -> Iterator[Optional[DialogueStateTracker]]:
        """Loads trackers according to strategy."""
        stored_keys = list(self.tracker_store.keys())
        if self.count is not None and self.count > len(stored_keys):
            rasa.shared.utils.io.raise_warning(
                "'count' exceeds number of trackers in the store"
            )
            self.count = len(stored_keys)

        keys = self.strategy(stored_keys, self.count)
        for sender in keys:
            yield self.tracker_store.retrieve(sender)
