from pathlib import Path
from typing import Dict, Text, Any, Optional

import logging
import tensorflow as tf
from tqdm import tqdm

import rasa.shared.utils.io

logger = logging.getLogger(__name__)


class RasaTrainingLogger(tf.keras.callbacks.Callback):
    """Callback for logging the status of training."""

    def __init__(self, epochs: int, silent: bool) -> None:
        """Initializes the callback.

        Args:
            epochs: Total number of epochs.
            silent: If 'True' the entire progressbar wrapper is disabled.
        """
        super().__init__()

        disable = silent or rasa.shared.utils.io.is_logging_disabled()
        self.progress_bar = tqdm(range(epochs), desc="Epochs", disable=disable)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Updates the logging output on every epoch end.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(logs)

    def on_train_end(self, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Closes the progress bar after training.

        Args:
            logs: The training metrics.
        """
        self.progress_bar.close()


class RasaModelCheckpoint(tf.keras.callbacks.Callback):
    """Callback for saving intermediate model checkpoints."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initializes the callback.

        Args:
            checkpoint_dir: Directory to store checkpoints to.
        """
        super().__init__()

        self.checkpoint_file = checkpoint_dir / "checkpoint.tf_model"
        self.best_metrics_so_far: Optional[Dict[Text, Any]] = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None:
        """Save the model on epoch end if the model has improved.

        Args:
            epoch: The current epoch.
            logs: The training metrics.
        """
        if self._does_model_improve(logs):
            logger.debug(f"Creating model checkpoint at epoch={epoch + 1} ...")
            self.model.save_weights(
                self.checkpoint_file, overwrite=True, save_format="tf"
            )

    def _does_model_improve(self, current_results: Dict[Text, Any]) -> bool:
        # Initialize best_metrics_so_far with the first results
        if not self.best_metrics_so_far:
            keys = filter(
                lambda k: k.startswith("val")
                and (k.endswith("_acc") or k.endswith("_f1")),
                current_results.keys(),
            )
            if not keys:
                return False

            for key in keys:
                self.best_metrics_so_far[key] = float(current_results[key])
            return True

        all_improved = all(
            [
                float(current_results[key]) > self.best_metrics_so_far[key]
                for key in self.best_metrics_so_far.keys()
            ]
        )

        if all_improved:
            for key in self.best_metrics_so_far.keys():
                self.best_metrics_so_far[key] = float(current_results[key])

        return all_improved
