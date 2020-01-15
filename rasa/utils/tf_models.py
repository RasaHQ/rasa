import typing
import logging
from typing import (
    List,
    Optional,
    Text,
    Dict,
    Tuple,
    Union,
    Generator,
    Callable,
    Any,
    NamedTuple,
)
from tqdm import tqdm
from rasa.utils import train_utils
from rasa.utils.common import is_logging_disabled
import tensorflow as tf

logger = logging.getLogger(__name__)


class RasaModel(tf.keras.models.Model):

    def compile(self):
        raise NotImplemented

    @staticmethod
    def _update_postfix_dict(
            postfix_dict: Dict[Text, Text], metrics, prefix: Text = ""
    ) -> Dict[Text, Text]:
        for name, value in metrics.loss.items():
            postfix_dict[f"{prefix}{name}"] = f"{value:.3f}"
        for name, value in metrics.score.items():
            postfix_dict[f"{prefix}{name}"] = f"{value:.3f}"
        return postfix_dict

    def fit(self,
            epochs: int,
            batch_size: Union[List[int], int],
            evaluate_on_num_examples: int,
            evaluate_every_num_epochs: int,
            silent: bool = False,
            eager: bool = False,
            output_file: Optional[Text] = None,
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
            train_dataset_func = self.train_dataset
            eval_dataset_func = self.eval_dataset

            train_on_batch_func = self.train_on_batch
        else:
            # allows increasing batch size
            train_dataset_func = tf.function(self.train_dataset)
            eval_dataset_func = tf.function(self.eval_dataset)

            train_on_batch_func = tf.function(
                self.train_on_batch, input_signature=[train_dataset_func(tf_batch_size).element_spec]
            )

        if evaluate_on_num_examples > 0:
            if eager:
                eval_func = self.eval
            else:
                eval_func = tf.function(
                    self.eval, input_signature=[eval_dataset_func(tf_batch_size).element_spec]
                )
        else:
            eval_func = None

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

            # Get the metric results
            postfix_dict = {k: v.result().numpy() for k, v in self.train_metrics.items()}

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
                        eval_func(batch_in)

                # Get the metric results
                postfix_dict.update(
                    {k: v.result().numpy() for k, v in self.eval_metrics.items()}
                )

            pbar.set_postfix(postfix_dict)

            # _write_training_metrics(output_file, ep, train_metrics, val_metrics)
        if not disable:
            logger.info("Finished training.")

    def evaluate(self):
        pass

    def predict(self):
        pass

    def train_on_batch(self, batch_in):
        raise NotImplementedError

    def test_on_batch(self):
        raise NotImplemented

    def predict_on_batch(self):
        raise NotImplemented

    def fit_generator(self):
        raise NotImplemented

    def evaluate_generator(self):
        raise NotImplemented

    def predict_generator(self):
        raise NotImplemented