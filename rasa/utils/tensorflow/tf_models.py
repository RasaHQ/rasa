import tensorflow as tf
import numpy as np
import logging
from collections import defaultdict
from typing import List, Text, Dict, Tuple, Union, Optional
from tqdm import tqdm
from rasa.utils.common import is_logging_disabled
from rasa.utils.tensorflow.tf_model_data import RasaModelData, FeatureSignature

logger = logging.getLogger(__name__)


# noinspection PyMethodOverriding
class RasaModel(tf.keras.models.Model):
    """Completely override all public methods of keras Model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics = {}

    def fit(
        self,
        model_data: RasaModelData,
        epochs: int,
        batch_size: Union[List[int], int],
        evaluate_on_num_examples: int,
        evaluate_every_num_epochs: int,
        batch_strategy: Text,
        silent: bool = False,
        eager: bool = True,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Train tf graph"""

        if evaluate_on_num_examples > 0:
            logger.info(
                f"Validation accuracy is calculated every {evaluate_every_num_epochs} "
                f"epochs."
            )

            model_data, evaluation_model_data = model_data.split(
                evaluate_on_num_examples, random_seed
            )

        disable = silent or is_logging_disabled()
        pbar = tqdm(range(epochs), desc="Epochs", disable=disable)

        tf_batch_size = tf.ones((), tf.int32)

        def train_dataset_function(_batch_size):
            return model_data.as_tf_dataset(_batch_size, batch_strategy, shuffle=True)

        def evaluation_dataset_function(_batch_size):
            return evaluation_model_data.as_tf_dataset(
                _batch_size, batch_strategy, shuffle=False
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
                tf_evaluation_on_batch_function = self.evaluate_on_batch
            else:
                tf_evaluation_dataset_function = tf.function(
                    func=evaluation_dataset_function
                )
                tf_evaluation_on_batch_function = tf.function(
                    self.evaluate_on_batch,
                    input_signature=[tf_evaluation_dataset_function(1).element_spec],
                )
        else:
            tf_evaluation_dataset_function = None
            tf_evaluation_on_batch_function = None

        for ep in pbar:
            ep_batch_size = tf_batch_size * self.linearly_increasing_batch_size(
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
                k: f"{v.result().numpy():3f}" for k, v in self.train_metrics.items()
            }

            if evaluate_on_num_examples > 0:
                if (
                    ep == 0
                    or (ep + 1) % evaluate_every_num_epochs == 0
                    or (ep + 1) == epochs
                ):
                    # Reset the metrics
                    for metric in self.train_metrics.values():
                        metric.reset_states()

                    # Eval on batches
                    self.set_training_phase(False)
                    for batch_in in tf_evaluation_dataset_function(ep_batch_size):
                        tf_evaluation_on_batch_function(batch_in)

                # Get the metric results
                postfix_dict.update(
                    {
                        f"val_{k}": f"{v.result().numpy():3f}"
                        for k, v in self.train_metrics.items()
                    }
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
        with tf.GradientTape() as tape:
            self._train_losses_scores(batch_in)
            regularization_loss = tf.math.add_n(self.losses)
            pred_loss = 0  # tf.math.add_n(list(losses.values()))
            total_loss = pred_loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_metrics["t_loss"].update_state(total_loss)

    def evaluate_on_batch(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]], **kwargs
    ) -> None:
        self._train_losses_scores(batch_in)
        regularization_loss = tf.math.add_n(self.losses)
        pred_loss = 0  # tf.math.add_n(list(losses.values()))
        total_loss = pred_loss + regularization_loss

        self.train_metrics["t_loss"].update_state(total_loss)

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

    @staticmethod
    def batch_to_model_data_format(
        batch: Union[Tuple[np.ndarray], Tuple[tf.Tensor]],
        data_signature: Dict[Text, List[FeatureSignature]],
    ) -> Dict[Text, List[tf.Tensor]]:
        """Convert input batch tensors into batch data format.
    
        Batch contains any number of batch data. The order is equal to the
        key-value pairs in session data. As sparse data were converted into indices, data,
        shape before, this methods converts them into sparse tensors. Dense data is
        kept.
        """

        batch_data = defaultdict(list)

        idx = 0
        for k, signature in data_signature.items():
            for is_sparse, shape in signature:
                if is_sparse:
                    # explicitly substitute last dimension in shape with known
                    # static value
                    batch_data[k].append(
                        tf.SparseTensor(
                            batch[idx],
                            batch[idx + 1],
                            [batch[idx + 2][0], batch[idx + 2][1], shape[-1]],
                        )
                    )
                    idx += 3
                else:
                    batch_data[k].append(batch[idx])
                    idx += 1

        return batch_data

    @staticmethod
    def linearly_increasing_batch_size(
        epoch: int, batch_size: Union[List[int], int], epochs: int
    ) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489.
        """

        if not isinstance(batch_size, list):
            return int(batch_size)

        if epochs > 1:
            return int(
                batch_size[0] + epoch * (batch_size[1] - batch_size[0]) / (epochs - 1)
            )
        else:
            return int(batch_size[0])
