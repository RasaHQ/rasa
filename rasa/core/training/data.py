from typing import List, Optional, Tuple

import numpy as np


# noinspection PyPep8Naming
class DialogueTrainingData:
    def __init__(
        self, X: np.ndarray, y: np.ndarray, true_length: Optional[List[int]] = None
    ) -> None:
        self.X = X
        self.y = y
        self.true_length = true_length

    def limit_training_data_to(self, max_samples: int) -> None:
        self.X = self.X[:max_samples]
        self.y = self.y[:max_samples]
        self.true_length = (
            self.true_length[:max_samples] if self.true_length is not None else None
        )

    def is_empty(self) -> bool:
        """Check if the training matrix does contain training samples."""
        return self.X.shape[0] == 0

    def max_history(self) -> int:
        return self.X.shape[1]

    def num_examples(self) -> int:
        return len(self.y)

    def shuffled_X_y(self) -> Tuple[np.ndarray, np.ndarray]:
        import numpy as np

        idx = np.arange(self.num_examples())
        np.random.shuffle(idx)
        shuffled_X = self.X[idx]
        shuffled_y = self.y[idx]
        return shuffled_X, shuffled_y
