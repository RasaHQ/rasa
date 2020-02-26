import tensorflow as tf
import numpy as np
import logging
from collections import defaultdict
from typing import List, Text, Dict, Tuple, Union, Optional, Callable
from tqdm import tqdm
from rasa.utils.common import is_logging_disabled
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
from rasa.utils.tensorflow.constants import SEQUENCE

logger = logging.getLogger(__name__)


# noinspection PyMethodOverriding
class RasaModel(tf.keras.models.Model):
    """Completely override all public methods of keras Model.

    Cannot be used as tf.keras.Model
    """

    def __init__(self, random_seed: Optional[int] = None, **kwargs) -> None:
        """Initialize the RasaModel.

        Args:
            random_seed: set the random seed to get reproducible results
        """
        super().__init__(**kwargs)

        self.total_loss = tf.keras.metrics.Mean(name="t_loss")
        self.metrics_to_log = ["t_loss"]

        self._training = None  # training phase should be defined when building a graph

        self._predict_function = None

        self.random_seed = random_seed

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        raise NotImplementedError

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        raise NotImplementedError

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
    ) -> None:
        """Fit model data"""

        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

        disable = silent or is_logging_disabled()

        evaluation_model_data = None
        if evaluate_on_num_examples > 0:
            if not disable:
                logger.info(
                    f"Validation accuracy is calculated every "
                    f"{evaluate_every_num_epochs} epochs."
                )

            model_data, evaluation_model_data = model_data.split(
                evaluate_on_num_examples, self.random_seed
            )

        (
            train_dataset_function,
            tf_train_on_batch_function,
        ) = self._get_tf_train_functions(eager, model_data, batch_strategy)
        (
            evaluation_dataset_function,
            tf_evaluation_on_batch_function,
        ) = self._get_tf_evaluation_functions(eager, evaluation_model_data)

        val_results = {}  # validation is not performed every epoch
        progress_bar = tqdm(range(epochs), desc="Epochs", disable=disable)

        for epoch in progress_bar:
            epoch_batch_size = self.linearly_increasing_batch_size(
                epoch, batch_size, epochs
            )

            self._batch_loop(
                train_dataset_function,
                tf_train_on_batch_function,
                epoch_batch_size,
                True,
            )

            postfix_dict = self._get_metric_results()

            if evaluate_on_num_examples > 0:
                if self._should_evaluate(evaluate_every_num_epochs, epochs, epoch):
                    self._batch_loop(
                        evaluation_dataset_function,
                        tf_evaluation_on_batch_function,
                        epoch_batch_size,
                        False,
                    )
                    val_results = self._get_metric_results(prefix="val_")

                postfix_dict.update(val_results)

            progress_bar.set_postfix(postfix_dict)

        self._training = None  # training phase should be defined when building a graph
        if not disable:
            logger.info("Finished training.")

    def train_on_batch(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> None:
        """Train on batch"""

        with tf.GradientTape() as tape:
            total_loss = self._total_batch_loss(batch_in)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def build_for_predict(
        self, predict_data: RasaModelData, eager: bool = False
    ) -> None:
        self._training = False  # needed for tf graph mode
        self._predict_function = self._get_tf_call_model_function(
            predict_data.as_tf_dataset, self.batch_predict, eager, "prediction"
        )

    def predict(self, predict_data: RasaModelData) -> Dict[Text, tf.Tensor]:
        if self._predict_function is None:
            logger.debug("There is no tensorflow prediction graph.")
            self.build_for_predict(predict_data)

        predict_dataset = predict_data.as_tf_dataset(batch_size=1)
        batch_in = next(iter(predict_dataset))

        self._training = False  # needed for eager mode
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
            batch_strategy=SEQUENCE,
            silent=True,  # don't confuse users with training output
            eager=True,  # no need to build tf graph, eager is faster here
        )
        # load trained weights
        model.load_weights(model_file_name)

        logger.debug("Finished loading the model.")
        return model

    def _total_batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
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
        batch_size: int,
        training: bool,
    ) -> None:
        """Run on batches"""

        self.reset_metrics()
        self._training = training  # needed for eager mode
        for batch_in in dataset_function(batch_size):
            call_model_function(batch_in)

    @staticmethod
    def _get_tf_call_model_function(
        dataset_function: Callable,
        call_model_function: Callable,
        eager: bool,
        phase: Text,
    ) -> Callable:
        """Convert functions to tensorflow functions"""

        if eager:
            return call_model_function

        logger.debug(f"Building tensorflow {phase} graph...")

        init_dataset = dataset_function(1)
        tf_call_model_function = tf.function(
            call_model_function, input_signature=[init_dataset.element_spec]
        )
        tf_call_model_function(next(iter(init_dataset)))

        logger.debug(f"Finished building tensorflow {phase} graph.")

        return tf_call_model_function

    def _get_tf_train_functions(
        self, eager: bool, model_data: RasaModelData, batch_strategy: Text
    ) -> Tuple[Callable, Callable]:
        """Create train tensorflow functions"""

        def train_dataset_function(_batch_size: int) -> tf.data.Dataset:
            return model_data.as_tf_dataset(_batch_size, batch_strategy, shuffle=True)

        self._training = True  # needed for tf graph mode
        return (
            train_dataset_function,
            self._get_tf_call_model_function(
                train_dataset_function, self.train_on_batch, eager, "train"
            ),
        )

    def _get_tf_evaluation_functions(
        self, eager: bool, evaluation_model_data: Optional[RasaModelData]
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        """Create evaluation tensorflow functions"""

        if evaluation_model_data is None:
            return None, None

        def evaluation_dataset_function(_batch_size: int) -> tf.data.Dataset:
            return evaluation_model_data.as_tf_dataset(
                _batch_size, SEQUENCE, shuffle=False
            )

        self._training = False  # needed for tf graph mode
        return (
            evaluation_dataset_function,
            self._get_tf_call_model_function(
                evaluation_dataset_function, self._total_batch_loss, eager, "evaluation"
            ),
        )

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
        batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]],
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
                    if isinstance(batch[idx], tf.Tensor):
                        batch_data[k].append(batch[idx])
                    else:
                        # convert to Tensor
                        batch_data[k].append(tf.constant(batch[idx], dtype=tf.float32))
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
        raise Exception(
            "This method should neither be called nor implemented in our code."
        )

    def evaluate(self, *args, **kwargs) -> None:
        raise Exception(
            "This method should neither be called nor implemented in our code."
        )

    def test_on_batch(self, *args, **kwargs) -> None:
        raise Exception(
            "This method should neither be called nor implemented in our code."
        )

    def predict_on_batch(self, *args, **kwargs) -> None:
        raise Exception(
            "This method should neither be called nor implemented in our code."
        )

    def fit_generator(self, *args, **kwargs) -> None:
        raise Exception(
            "This method should neither be called nor implemented in our code."
        )

    def evaluate_generator(self, *args, **kwargs) -> None:
        raise Exception(
            "This method should neither be called nor implemented in our code."
        )

    def predict_generator(self, *args, **kwargs) -> None:
        raise Exception(
            "This method should neither be called nor implemented in our code."
        )
