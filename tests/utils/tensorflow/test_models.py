import pytest
from typing import Dict, Text, Union, Tuple
import numpy as np
import tensorflow as tf

from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.model_data import FeatureArray
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.utils.tensorflow.constants import (
    LABEL,
    IDS,
    SENTENCE,
    RANDOM_SEED,
    REGULARIZATION_CONSTANT,
    CONNECTION_DENSITY,
    DENSE_DIMENSION,
    LEARNING_RATE,
    SEQUENCE,
    SPARSE_INPUT_DROPOUT,
    DROP_RATE,
    DENSE_INPUT_DROPOUT,
    HIDDEN_LAYERS_SIZES,
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    TRANSFORMER_SIZE,
    UNIDIRECTIONAL_ENCODER,
    DROP_RATE_ATTENTION,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
import scipy.sparse
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.utils.tensorflow import rasa_layers
from tensorflow.python.framework.errors_impl import InvalidArgumentError


@pytest.mark.parametrize(
    "existing_outputs, new_batch_outputs, expected_output",
    [
        (
            {"a": np.array([1, 2]), "b": np.array([3, 1])},
            {"a": np.array([5, 6]), "b": np.array([2, 4])},
            {"a": np.array([1, 2, 5, 6]), "b": np.array([3, 1, 2, 4])},
        ),
        (
            {},
            {"a": np.array([5, 6]), "b": np.array([2, 4])},
            {"a": np.array([5, 6]), "b": np.array([2, 4])},
        ),
        (
            {"a": np.array([1, 2]), "b": {"c": np.array([3, 1])}},
            {"a": np.array([5, 6]), "b": {"c": np.array([2, 4])}},
            {"a": np.array([1, 2, 5, 6]), "b": {"c": np.array([3, 1, 2, 4])}},
        ),
    ],
)
def test_merging_batch_outputs(
    existing_outputs: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
    new_batch_outputs: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
    expected_output: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
):

    predicted_output = RasaModel._merge_batch_outputs(
        existing_outputs, new_batch_outputs
    )

    def test_equal_dicts(
        dict1: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
        dict2: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
    ) -> None:
        assert dict2.keys() == dict1.keys()
        for key in dict1:
            val_1 = dict1[key]
            val_2 = dict2[key]
            assert type(val_1) == type(val_2)

            if isinstance(val_2, np.ndarray):
                assert np.array_equal(val_1, val_2)

            elif isinstance(val_2, dict):
                test_equal_dicts(val_1, val_2)

    test_equal_dicts(predicted_output, expected_output)


@pytest.mark.parametrize(
    "batch_size, number_of_data_points, expected_number_of_batch_iterations",
    [(2, 3, 2), (1, 3, 3), (5, 3, 1),],
)
def test_batch_inference(
    batch_size: int,
    number_of_data_points: int,
    expected_number_of_batch_iterations: int,
):
    model = RasaModel()

    def _batch_predict(
        batch_in: Tuple[np.ndarray],
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]]:

        dummy_output = batch_in[0]
        output = {
            "dummy_output": dummy_output,
            "non_input_affected_output": tf.constant(
                np.array([[1, 2]]), dtype=tf.int32
            ),
        }
        return output

    # Monkeypatch batch predict so that run_inference interface can be tested
    model.batch_predict = _batch_predict

    # Create dummy model data to pass to model
    model_data = RasaModelData(
        label_key=LABEL,
        label_sub_key=IDS,
        data={
            TEXT: {
                SENTENCE: [
                    FeatureArray(
                        np.random.rand(number_of_data_points, 2),
                        number_of_dimensions=2,
                    ),
                ]
            }
        },
    )
    output = model.run_inference(model_data, batch_size=batch_size)

    # Firstly, the number of data points in dummy_output should be equal
    # to the number of data points sent as input.
    assert output["dummy_output"].shape[0] == number_of_data_points

    # Secondly, the number of data points inside diagnostic_data should be
    # equal to the number of batches passed to the model because for every
    # batch passed as input, it would have created a
    # corresponding diagnostic data entry.
    assert output["non_input_affected_output"].shape == (
        expected_number_of_batch_iterations,
        2,
    )


def test_adjusting_sparse_layers(monkeypatch):
    """Tests adjusting sizes of `DenseForSparse` layers inside
    `TransformerRasaModel` object."""

    def mock_check_data(self):
        pass

    monkeypatch.setattr(TransformerRasaModel, "_check_data", mock_check_data)
    # in this case, we have only one sparse featurizer
    initial_sparse_feature_size = 4
    final_sparse_feature_size = 5
    config = {
        REGULARIZATION_CONSTANT: 0.02,
        LEARNING_RATE: 0.01,
        RANDOM_SEED: 10,
        DENSE_DIMENSION: {TEXT: 128, LABEL: 20},
        SPARSE_INPUT_DROPOUT: True,
        DROP_RATE: 0.2,
        DENSE_INPUT_DROPOUT: True,
        HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []},
        CONNECTION_DENSITY: 0.2,
        MASKED_LM: False,
        NUM_TRANSFORMER_LAYERS: 2,
        NUM_HEADS: 4,
        TRANSFORMER_SIZE: 256,
        UNIDIRECTIONAL_ENCODER: False,
        DROP_RATE_ATTENTION: 0,
        KEY_RELATIVE_ATTENTION: False,
        VALUE_RELATIVE_ATTENTION: False,
        MAX_RELATIVE_POSITION: None,
    }
    message = Message.build(text="hi how are you?", intent="intent_name")
    sentence_features = Features(
        features=scipy.sparse.csr_matrix([1, 1, 1, 1]),
        feature_type=FEATURE_TYPE_SENTENCE,
        attribute=TEXT,
        origin="origin",
    )
    sequence_features = Features(
        features=scipy.sparse.csr_matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        feature_type=FEATURE_TYPE_SEQUENCE,
        attribute=TEXT,
        origin="origin",
    )
    label_features = Features(
        features=scipy.sparse.csr_matrix([1]),
        feature_type=FEATURE_TYPE_SEQUENCE,
        attribute=INTENT,
        origin="origin",
    )
    message.add_features(sequence_features)
    message.add_features(sentence_features)
    message.add_features(label_features)
    diet_classifier = DIETClassifier()
    model_data = diet_classifier._create_model_data(
        training_data=[message],
        label_id_dict={"intent_name": 0},
        label_attribute=INTENT,
    )
    old_sparse_feature_sizes = model_data.get_sparse_feature_sizes()
    label_data = RasaModelData()
    sequence, _ = message.get_sparse_features(INTENT)
    sequence_features = [
        FeatureArray(np.array([sequence.features]), number_of_dimensions=3)
    ]
    label_data.add_features(LABEL, SEQUENCE, sequence_features)
    model = TransformerRasaModel(
        data_signature=model_data.get_signature(),
        label_data=label_data,
        config=config,
        name="model_name",
    )
    model._tf_layers[f"sequence_layer.{TEXT}"] = rasa_layers.RasaSequenceLayer(
        TEXT, model.data_signature[TEXT], model.config
    )
    new_message = Message.build(text="good", intent="intent_name")
    new_sentence_features = Features(
        features=scipy.sparse.csr_matrix([0, 0, 0, 0, 1]),
        feature_type=FEATURE_TYPE_SENTENCE,
        attribute=TEXT,
        origin="origin",
    )
    new_sequence_features = Features(
        features=scipy.sparse.csr_matrix([[0, 0, 0, 0, 1]]),
        feature_type=FEATURE_TYPE_SEQUENCE,
        attribute=TEXT,
        origin="origin",
    )
    new_label_features = Features(
        features=scipy.sparse.csr_matrix([1]),
        feature_type=FEATURE_TYPE_SEQUENCE,
        attribute=INTENT,
        origin="origin",
    )
    new_message.add_features(new_sequence_features)
    new_message.add_features(new_sentence_features)
    new_message.add_features(new_label_features)
    new_model_data = diet_classifier._create_model_data(
        training_data=[new_message],
        label_id_dict={"intent_name": 0},
        label_attribute=INTENT,
    )

    new_sparse_feature_sizes = new_model_data.get_sparse_feature_sizes()
    model.compile(optimizer=tf.keras.optimizers.Adam(model.config[LEARNING_RATE]))
    layer_input = tf.sparse.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=[1, 2],
        dense_shape=[2, initial_sparse_feature_size],
    )
    layers = model._tf_layers["sequence_layer.text"]._tf_layers["feature_combining"]
    initial_sequence_layer = layers._tf_layers["sparse_dense.sequence"]._tf_layers[
        "sparse_to_dense"
    ]
    initial_sentence_layer = layers._tf_layers["sparse_dense.sentence"]._tf_layers[
        "sparse_to_dense"
    ]
    try:
        initial_sentence_layer(layer_input)
    except InvalidArgumentError:
        print("this needs to be fixed")
    try:
        initial_sequence_layer(layer_input)
    except InvalidArgumentError:
        print("this needs to be fixed")
    model._update_dense_for_sparse_layers(
        new_sparse_feature_sizes, old_sparse_feature_sizes
    )
    new_layer_input = tf.sparse.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=[1, 2],
        dense_shape=[2, final_sparse_feature_size],
    )
    final_sequence_layer = layers._tf_layers["sparse_dense.sequence"]._tf_layers[
        "sparse_to_dense"
    ]
    final_sentence_layer = layers._tf_layers["sparse_dense.sentence"]._tf_layers[
        "sparse_to_dense"
    ]
    try:
        final_sentence_layer(new_layer_input)
    except InvalidArgumentError:
        print("this needs to be fixed")
    try:
        final_sequence_layer(new_layer_input)
    except InvalidArgumentError:
        print("this needs to be fixed")
    # test size of output units
    assert initial_sequence_layer.get_units() == initial_sentence_layer.get_units()
    initial_output_units = initial_sequence_layer.get_units()
    assert final_sequence_layer.get_units() == final_sentence_layer.get_units()
    final_output_units = final_sequence_layer.get_units()
    assert initial_output_units == final_output_units
    # test `sparse_feature_sizes` collections
    assert (
        old_sparse_feature_sizes[TEXT][FEATURE_TYPE_SENTENCE][0]
        == initial_sparse_feature_size
    )
    assert (
        old_sparse_feature_sizes[TEXT][FEATURE_TYPE_SEQUENCE][0]
        == initial_sparse_feature_size
    )
    assert (
        new_sparse_feature_sizes[TEXT][FEATURE_TYPE_SENTENCE][0]
        == final_sparse_feature_size
    )
    assert (
        new_sparse_feature_sizes[TEXT][FEATURE_TYPE_SEQUENCE][0]
        == final_sparse_feature_size
    )
    # test kernel shapes
    initial_sequence_kernel = initial_sequence_layer.get_kernel()
    initial_sentence_kernel = initial_sentence_layer.get_kernel()
    final_sequence_kernel = final_sequence_layer.get_kernel()
    final_sentence_kernel = final_sentence_layer.get_kernel()
    assert initial_sentence_kernel.shape == (
        initial_sparse_feature_size,
        initial_output_units,
    )
    assert initial_sequence_kernel.shape == (
        initial_sparse_feature_size,
        initial_output_units,
    )
    assert final_sentence_kernel.shape == (
        final_sparse_feature_size,
        final_output_units,
    )
    assert final_sequence_kernel.shape == (
        final_sparse_feature_size,
        final_output_units,
    )
    # test if the final kernel contains initial one
    assert np.array_equal(
        final_sentence_kernel[:initial_sparse_feature_size, :],
        initial_sentence_kernel[:initial_sparse_feature_size, :],
    )
    assert np.array_equal(
        final_sequence_kernel[:initial_sparse_feature_size, :],
        initial_sequence_kernel[:initial_sparse_feature_size, :],
    )
