from typing import Text, List
import pickle
import os

from rasa.shared.nlu.training_data.features import Features


class FeaturePersistence:
    MSG_PER_CHUNK = 512


class FeatureWriter(FeaturePersistence):
    """Writes features to disc."""

    def __init__(self, path: Text):
        self.path = path
        self._maybe_create_path()
        self.number_written = 0
        self._current_features = []
        self._file_idx = 0

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._flush()

    def _maybe_create_path(self) -> None:
        """creates target path if necessary."""
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def write(self, features: List[Features]) -> None:
        """Caches the features and maybe writes them to disc."""
        self._current_features.append(features)
        self.number_written += 1
        if self.number_written % self.MSG_PER_CHUNK == 0:
            self._flush()

    def _flush(self) -> None:
        """Flushes the current messages to disc."""
        target_file = f"{self.path}/features_{self._file_idx:03}.pickle"
        with open(target_file, "wb") as out:
            pickle.dump(self._current_features, out)

        self._current_features = []
        self._file_idx += 1


class FeatureReader(FeaturePersistence):
    """Reads features from disc."""

    def __init__(self, path: Text):
        self.path = path
        self._file_idx = 0
        self._current_features = []

    def __iter__(self):
        self._file_idx = 0
        self._load_current_chunk()
        return self

    def __next__(self) -> List[Features]:
        if self._current_features:
            return self._current_features.pop()
        else:
            self._file_idx += 1
            try:
                self._load_current_chunk()
            except:
                raise StopIteration
            else:
                return next(self)

    def _load_current_chunk(self):
        target_file = f"{self.path}/features_{self._file_idx:03}.pickle"
        with open(target_file, "rb") as f:
            self._current_features = pickle.load(f)[::-1]
