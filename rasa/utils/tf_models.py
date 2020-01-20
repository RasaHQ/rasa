import numpy as np
import logging
from typing import List, Text, Dict, Tuple, Union
from tqdm import tqdm
from rasa.utils import train_utils
from rasa.utils.common import is_logging_disabled
import tensorflow as tf

from rasa.utils.tf_model_data import RasaModelData

logger = logging.getLogger(__name__)


class RasaModel(tf.keras.models.Model):
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
        model_data: RasaModelData,
        epochs: int,
        batch_size: Union[List[int], int],
        evaluate_on_num_examples: int,
        evaluate_every_num_epochs: int,
        label_key: Text,
        batch_strategy: Text,
        silent: bool = False,
        eager: bool = False,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        """Train tf graph"""

        if evaluate_on_num_examples > 0:
            logger.info(
                f"Validation accuracy is calculated every {evaluate_every_num_epochs} "
                f"epochs."
            )

            model_data, evaluation_model_data = model_data.split(
                evaluate_on_num_examples, random_seed, label_key=label_key
            )

        disable = silent or is_logging_disabled()
        pbar = tqdm(range(epochs), desc="Epochs", disable=disable)

        tf_batch_size = tf.ones((), tf.int32)

        def train_dataset_function(x):
            return model_data.as_tf_dataset(x, label_key, batch_strategy, shuffle=True)

        def evaluation_dataset_function(x):
            return evaluation_model_data.as_tf_dataset(
                x, label_key, batch_strategy, shuffle=False
            )

        if eager:
            # allows increasing batch size
            tf_train_dataset_function = train_dataset_function
            tf_train_on_batch_function = self.train_on_batch
        else:
            # allows increasing batch size
            tf_train_dataset_function = tf.function(func=train_dataset_function)
            tf_train_on_batch_function = tf.function(
                self.train_on_batch,
                input_signature=[tf_train_dataset_function(1).element_spec],
            )

        if evaluate_on_num_examples > 0:
            if eager:
                tf_evaluation_dataset_function = evaluation_dataset_function
                tf_evaluation_function = self.eval
            else:
                tf_evaluation_dataset_function = tf.function(
                    func=evaluation_dataset_function
                )
                tf_evaluation_function = tf.function(
                    self.eval,
                    input_signature=[tf_evaluation_dataset_function(1).element_spec],
                )
        else:
            tf_evaluation_dataset_function = None
            tf_evaluation_function = None

        for ep in pbar:
            ep_batch_size = tf_batch_size * train_utils.linearly_increasing_batch_size(
                ep, batch_size, epochs
            )

            # Reset the metrics
            for metric in self.train_metrics.values():
                metric.reset_states()

            # Train on batches
            self.set_training_phase(True)
            for batch_in in tf_train_dataset_function(ep_batch_size):
                tf_train_on_batch_function(batch_in)

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
                    for batch_in in tf_evaluation_dataset_function(ep_batch_size):
                        tf_evaluation_function(batch_in)

                # Get the metric results
                postfix_dict.update(
                    {k: v.result().numpy() for k, v in self.eval_metrics.items()}
                )

            pbar.set_postfix(postfix_dict)

        if not disable:
            logger.info("Finished training.")

    def compile(self, **kwargs) -> None:
        raise NotImplementedError

    def evaluate(self, **kwargs) -> None:
        pass

    def predict(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]], **kwargs
    ) -> Dict[Text, tf.Tensor]:
        pass

    def train_on_batch(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]], **kwargs
    ) -> None:
        raise NotImplementedError

    def test_on_batch(self, **kwargs) -> None:
        raise NotImplementedError

    def predict_on_batch(self, **kwargs) -> None:
        raise NotImplementedError

    def fit_generator(self, **kwargs) -> None:
        raise NotImplementedError

    def evaluate_generator(self, **kwargs) -> None:
        raise NotImplementedError

    def predict_generator(self, **kwargs) -> None:
        raise NotImplementedError
