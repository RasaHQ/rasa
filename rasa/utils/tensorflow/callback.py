import logging
from pathlib import Path
from typing import Any, Dict, Optional, Text

from rasa.utils.installation_utils import check_for_installation_issues
from rasa.utils.tensorflow import TENSORFLOW_AVAILABLE

if TENSORFLOW_AVAILABLE:
    check_for_installation_issues()
    import tensorflow as tf
    from tqdm import tqdm
else:
    # Placeholder values when TensorFlow is not available
    tf = None
    tqdm = None

import rasa.shared.utils.io

logger = logging.getLogger(__name__)


if TENSORFLOW_AVAILABLE:

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

        def on_epoch_end(
            self, epoch: int, logs: Optional[Dict[Text, Any]] = None
        ) -> None:
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

            self.checkpoint_file = checkpoint_dir / "checkpoint.weights.h5"
            self.best_metrics_so_far: Dict[Text, Any] = {}

        def on_epoch_end(
            self, epoch: int, logs: Optional[Dict[Text, Any]] = None
        ) -> None:
            """Save the model on epoch end if the model has improved.

            Args:
                epoch: The current epoch.
                logs: The training metrics.
            """
            if self._does_model_improve(logs):
                logger.debug(f"Creating model checkpoint at epoch={epoch + 1} ...")
                # Ensure model is built before saving weights
                if not self.model.built:
                    # Build the model with dummy data to ensure it's built
                    import tensorflow as tf

                    dummy_input = tf.zeros((1, 1))
                    _ = self.model(dummy_input)

                # Ensure the directory exists before saving
                import os

                os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
                self.model.save_weights(self.checkpoint_file, overwrite=True)

        def _does_model_improve(self, curr_results: Dict[Text, Any]) -> bool:
            """Checks whether the current results are better than the best so far.

            Results are considered better if each metric is
            equal or better than the best so far, and at least one is better.

            Args:
                curr_results: The training metrics for this epoch.
            """
            curr_metric_names = [
                k
                for k in curr_results.keys()
                if k.startswith("val") and (k.endswith("_acc") or k.endswith("_f1"))
            ]
            # the "val" prefix is prepended to metrics in fit
            # if _should_eval returns true
            # for this particular epoch
            if len(curr_metric_names) == 0:
                # the metrics are not validation metrics
                return False
            # initialize best_metrics_so_far with the first results
            if not self.best_metrics_so_far:
                for metric_name in curr_metric_names:
                    self.best_metrics_so_far[metric_name] = float(
                        curr_results[metric_name]
                    )
                    return True

            at_least_one_improved = False
            improved_metrics = {}
            for metric_name in self.best_metrics_so_far.keys():
                if (
                    float(curr_results[metric_name])
                    < self.best_metrics_so_far[metric_name]
                ):
                    # at least one of the values is worse
                    return False
                if (
                    float(curr_results[metric_name])
                    > self.best_metrics_so_far[metric_name]
                ):
                    at_least_one_improved = True
                    improved_metrics[metric_name] = float(curr_results[metric_name])

            # all current values >= previous best and at least one is better
            if at_least_one_improved:
                self.best_metrics_so_far.update(improved_metrics)
            return at_least_one_improved
else:
    # Placeholder classes when TensorFlow is not available
    RasaTrainingLogger = None  # type: ignore
    RasaModelCheckpoint = None  # type: ignore
