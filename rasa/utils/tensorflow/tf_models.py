import tensorflow as tf
import numpy as np
import logging
from collections import defaultdict
from typing import List, Text, Dict, Tuple, Union, Optional, Callable
from tqdm import tqdm
from rasa.utils.common import is_logging_disabled
from rasa.utils.tensorflow.tf_model_data import RasaModelData, FeatureSignature

logger = logging.getLogger(__name__)


# noinspection PyMethodOverriding
class RasaModel(tf.keras.models.Model):
    """Completely override all public methods of keras Model.

    Cannot be used as tf.keras.Model
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.total_loss = tf.keras.metrics.Mean(name="t_loss")
        self.metrics_to_log = ["t_loss"]

        self._training = tf.ones((), tf.bool)
        self._optimizer = None

        self._predict_function = None

    def batch_loss(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]]
    ) -> tf.Tensor:
        raise NotImplementedError

    def batch_predict(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]]
    ) -> Dict[Text, tf.Tensor]:
        raise NotImplementedError

    def set_training_phase(self, training: bool) -> None:
        if training:
            self._training = tf.ones((), tf.bool)
        else:
            self._training = tf.zeros((), tf.bool)

    def fit(
        self,
        model_data: RasaModelData,
        epochs: int,
        batch_size: Union[List[int], int],
        evaluate_on_num_examples: int,
        evaluate_every_num_epochs: int,
        batch_strategy: Text,
        silent: bool = False,
        eager: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """Fit model data"""

        disable = silent or is_logging_disabled()

        evaluation_model_data = None
        if evaluate_on_num_examples > 0:
            if not disable:
                logger.info(
                    f"Validation accuracy is calculated every "
                    f"{evaluate_every_num_epochs} epochs."
                )

            model_data, evaluation_model_data = model_data.split(
                evaluate_on_num_examples, random_seed
            )

        (
            tf_train_dataset_function,
            tf_train_on_batch_function,
        ) = self._get_tf_train_functions(eager, model_data, batch_strategy)

        (
            tf_evaluation_dataset_function,
            tf_evaluation_on_batch_function,
        ) = self._get_tf_evaluation_functions(
            eager, evaluate_on_num_examples, evaluation_model_data
        )

        pbar = tqdm(range(epochs), desc="Epochs", disable=disable)

        for ep in pbar:
            ep_batch_size = self.linearly_increasing_batch_size(ep, batch_size, epochs)
            if not eager:
                ep_batch_size *= tf.ones((), tf.int32)

            self._batch_loop(
                tf_train_dataset_function,
                tf_train_on_batch_function,
                ep_batch_size,
                True,
            )

            postfix_dict = self._get_metric_results()

            if evaluate_on_num_examples > 0:
                if self._should_evaluate(evaluate_every_num_epochs, epochs, ep):
                    self._batch_loop(
                        tf_evaluation_dataset_function,
                        tf_evaluation_on_batch_function,
                        ep_batch_size,
                        False,
                    )

                # Get the metric results
                postfix_dict.update(self._get_metric_results(prefix="val_"))

            pbar.set_postfix(postfix_dict)

        if not disable:
            logger.info("Finished training.")

    def train_on_batch(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]]
    ) -> None:
        """Train on batch"""

        with tf.GradientTape() as tape:
            total_loss = self._total_batch_loss(batch_in)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def build_for_predict(
        self, predict_data: RasaModelData, eager: bool = False
    ) -> None:
        def predict_dataset_function(  # to reuse the same helper method
            _batch_size: Union[tf.Tensor, int]
        ) -> tf.data.Dataset:
            return predict_data.as_tf_dataset(_batch_size, "sequence", shuffle=False)

        _, self._predict_function = self._get_tf_functions(
            predict_dataset_function, self.batch_predict, eager, "prediction"
        )

    def predict(self, predict_data: RasaModelData) -> Dict[Text, tf.Tensor]:
        if self._predict_function is None:
            logger.debug("There is no tensorflow prediction graph.")
            self.build_for_predict(predict_data)

        predict_dataset = predict_data.as_tf_dataset(batch_size=1)
        batch_in = next(iter(predict_dataset))
        self.set_training_phase(False)
        return self._predict_function(batch_in)

    def save(self, model_file_name: Text) -> None:
        self.save_weights(model_file_name, save_format="tf")

    @classmethod
    def load(
        cls, model_file_name: Text, model_data_example: RasaModelData, *args, **kwargs
    ) -> "RasaModel":
        logger.debug("Loading the model ...")
        # create empty model
        model = cls(*args, **kwargs)
        # need to train on 1 example to build weights of the correct size
        model.fit(
            model_data_example,
            epochs=1,
            batch_size=1,
            evaluate_every_num_epochs=0,
            evaluate_on_num_examples=0,
            batch_strategy="sequence",
            silent=True,  # don't confuse users with training output
            eager=True,  # no need to build tf graph, eager is faster here
        )
        # load trained weights
        model.load_weights(model_file_name)
        logger.debug("Finished loading the model.")
        return model

    def _total_batch_loss(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]]
    ) -> tf.Tensor:
        """Calculate total loss"""

        prediction_loss = self.batch_loss(batch_in)
        regularization_loss = tf.math.add_n(self.losses)
        total_loss = prediction_loss + regularization_loss
        self.total_loss.update_state(total_loss)

        return total_loss

    def _batch_loop(
        self,
        dataset_function: Callable,
        call_model_function: Callable,
        batch_size: Union[tf.Tensor, int],
        training: bool,
    ) -> None:
        """Run on batches"""

        self.reset_metrics()
        self.set_training_phase(training)
        for batch_in in dataset_function(batch_size):
            call_model_function(batch_in)

    @staticmethod
    def _get_tf_functions(
        dataset_function: Callable,
        call_model_function: Callable,
        eager: bool,
        phase: Text,
    ) -> Tuple[Callable, Callable]:
        """Convert functions to tensorflow functions"""

        if eager:
            return dataset_function, call_model_function

        logger.debug(f"Building tensorflow {phase} graph...")
        # allows increasing batch size
        tf_dataset_function = tf.function(func=dataset_function)

        init_dataset = tf_dataset_function(tf.ones((), tf.int32))

        tf_method_function = tf.function(
            call_model_function, input_signature=[init_dataset.element_spec]
        )
        tf_method_function(next(iter(init_dataset)))

        logger.debug(f"Finished building tensorflow {phase} graph.")

        return tf_dataset_function, tf_method_function

    def _get_tf_train_functions(
        self, eager: bool, model_data: RasaModelData, batch_strategy: Text
    ) -> Tuple[Callable, Callable]:
        """Create train tensorflow functions"""

        def train_dataset_function(
            _batch_size: Union[tf.Tensor, int]
        ) -> tf.data.Dataset:
            return model_data.as_tf_dataset(_batch_size, batch_strategy, shuffle=True)

        return self._get_tf_functions(
            train_dataset_function, self.train_on_batch, eager, "train"
        )

    def _get_tf_evaluation_functions(
        self,
        eager: bool,
        evaluate_on_num_examples: int,
        evaluation_model_data: RasaModelData,
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        """Create evaluation tensorflow functions"""

        if evaluate_on_num_examples > 0:

            def evaluation_dataset_function(
                _batch_size: Union[tf.Tensor, int]
            ) -> tf.data.Dataset:
                return evaluation_model_data.as_tf_dataset(
                    _batch_size, "sequence", shuffle=False
                )

            return self._get_tf_functions(
                evaluation_dataset_function, self._total_batch_loss, eager, "evaluation"
            )

        return None, None

    def _get_metric_results(self, prefix: Optional[Text] = None) -> Dict[Text, Text]:
        """Get the metrics results"""

        prefix = prefix or ""

        return {
            f"{prefix}{metric.name}": f"{metric.result().numpy():.3f}"
            for metric in self.metrics
            if metric.name in self.metrics_to_log
        }

    @staticmethod
    def _should_evaluate(
        evaluate_every_num_epochs: int, epochs: int, current_epoch: int
    ) -> bool:
        return (
            current_epoch == 0
            or (current_epoch + 1) % evaluate_every_num_epochs == 0
            or (current_epoch + 1) == epochs
        )

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

    def compile(self, *args, **kwargs) -> None:
        raise NotImplemented

    def evaluate(self, *args, **kwargs) -> None:
        raise NotImplemented

    def test_on_batch(self, *args, **kwargs) -> None:
        raise NotImplemented

    def predict_on_batch(self, *args, **kwargs) -> None:
        raise NotImplemented

    def fit_generator(self, *args, **kwargs) -> None:
        raise NotImplemented

    def evaluate_generator(self, *args, **kwargs) -> None:
        raise NotImplemented

    def predict_generator(self, *args, **kwargs) -> None:
        raise NotImplemented
