import numpy as np
import logging
from typing import List, Optional, Text, Dict, Tuple, Union
from tqdm import tqdm
from rasa.utils import train_utils
from rasa.utils.common import is_logging_disabled
import tensorflow as tf

from rasa.utils.train_utils import SessionDataType

logger = logging.getLogger(__name__)


# noinspection PyMethodOverriding
class RasaModel(tf.keras.models.Model):
    """Completely override all public methods of keras Model."""

    @staticmethod
    def _update_postfix_dict(
        postfix_dict: Dict[Text, Text], metrics, prefix: Text = ""
    ) -> Dict[Text, Text]:
        for name, value in metrics.loss.items():
            postfix_dict[f"{prefix}{name}"] = f"{value:.3f}"
        for name, value in metrics.score.items():
            postfix_dict[f"{prefix}{name}"] = f"{value:.3f}"
        return postfix_dict

    def fit(
        self,
        epochs: int,
        batch_size: Union[List[int], int],
        session_data: SessionDataType,
        eval_session_data: Optional[SessionDataType],
        evaluate_on_num_examples: int,
        evaluate_every_num_epochs: int,
        silent: bool = False,
        eager: bool = False,
    ) -> None:
        """Train tf graph"""

        if evaluate_on_num_examples > 0:
            logger.info(
                f"Validation accuracy is calculated every {evaluate_every_num_epochs} "
                f"epochs."
            )
        disable = silent or is_logging_disabled()
        pbar = tqdm(range(epochs), desc="Epochs", disable=disable)

        tf_batch_size = tf.ones((), tf.int32)

        if eager:
            # allows increasing batch size
            train_dataset_func = lambda x: self.train_dataset(x, session_data)
            train_on_batch_func = self.train_on_batch
        else:
            # allows increasing batch size
            train_dataset_func = tf.function(
                func=lambda x: self.train_dataset(x, session_data)
            )
            train_on_batch_func = tf.function(
                self.train_on_batch,
                input_signature=[train_dataset_func(1).element_spec],
            )

        if evaluate_on_num_examples > 0:
            if eager:
                eval_dataset_func = lambda x: self.eval_dataset(x, eval_session_data)
                evaluate_on_batch_func = self.evaluate_on_batch
            else:
                eval_dataset_func = tf.function(
                    func=lambda x: self.eval_dataset(x, eval_session_data)
                )
                evaluate_on_batch_func = tf.function(
                    self.evaluate_on_batch, input_signature=[eval_dataset_func(1).element_spec]
                )
        else:
            eval_dataset_func = None
            evaluate_on_batch_func = None

        for ep in pbar:
            ep_batch_size = tf_batch_size * train_utils.linearly_increasing_batch_size(
                ep, batch_size, epochs
            )

            # Reset the metrics
            for metric in self.train_metrics.values():
                metric.reset_states()

            # Train on batches
            self.set_training_phase(True)
            for batch_in in train_dataset_func(ep_batch_size):
                train_on_batch_func(batch_in)

            # print(self.metrics)
            # exit()

            # Get the metric results
            postfix_dict = {
                k: v.result().numpy() for k, v in self.train_metrics.items()
            }

            if evaluate_on_num_examples > 0:
                if (
                    ep == 0
                    or (ep + 1) % evaluate_every_num_epochs == 0
                    or (ep + 1) == epochs
                ):
                    # Reset the metrics
                    for metric in self.eval_metrics.values():
                        metric.reset_states()

                    # Eval on batches
                    self.set_training_phase(False)
                    for batch_in in eval_dataset_func(ep_batch_size):
                        evaluate_on_batch_func(batch_in)

                # Get the metric results
                postfix_dict.update(
                    {k: v.result().numpy() for k, v in self.eval_metrics.items()}
                )

            pbar.set_postfix(postfix_dict)

            # _write_training_metrics(output_file, ep, train_metrics, val_metrics)
        if not disable:
            logger.info("Finished training.")

    def compile(self) -> None:
        raise NotImplemented

    def evaluate(self) -> None:
        pass

    def predict(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]]
    ) -> Dict[Text, tf.Tensor]:
        pass

    def train_on_batch(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]]
    ) -> None:
        with tf.GradientTape() as tape:
            losses, scores = self._train_losses_scores(batch_in)
            regularization_loss = tf.math.add_n(self.losses)
            pred_loss = tf.math.add_n(list(losses.values()))
            total_loss = pred_loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_metrics["t_loss"].update_state(total_loss)
        for k, v in losses.items():
            self.train_metrics[k].update_state(v)
        for k, v in scores.items():
            self.train_metrics[k].update_state(v)

    def evaluate_on_batch(self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]]):
        losses, scores = self._train_losses_scores(batch_in)
        regularization_loss = tf.math.add_n(self.losses)
        pred_loss = tf.math.add_n(list(losses.values()))
        total_loss = pred_loss + regularization_loss

        self.eval_metrics["val_t_loss"].update_state(total_loss)
        for k, v in losses.items():
            self.eval_metrics[f"val_{k}"].update_state(v)
        for k, v in scores.items():
            self.eval_metrics[f"val_{k}"].update_state(v)

    def test_on_batch(self) -> None:
        raise NotImplemented

    def predict_on_batch(self) -> None:
        raise NotImplemented

    def fit_generator(self) -> None:
        raise NotImplemented

    def evaluate_generator(self) -> None:
        raise NotImplemented

    def predict_generator(self) -> None:
        raise NotImplemented
