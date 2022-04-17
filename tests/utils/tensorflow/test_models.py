import pytest
from typing import Dict, Text, Union, Tuple, List
import numpy as np
import tensorflow as tf

from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.model_data import FeatureArray
from rasa.utils.tensorflow.constants import LABEL, IDS, SENTENCE
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE


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
    [(2, 3, 2), (1, 3, 3), (5, 3, 1)],
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
                        np.random.rand(number_of_data_points, 2), number_of_dimensions=2
                    )
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


@pytest.mark.parametrize(
    "new_sparse_feature_sizes, old_sparse_feature_sizes, raise_exception",
    [
        (
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [], FEATURE_TYPE_SENTENCE: [1]},
            },
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [], FEATURE_TYPE_SENTENCE: [2]},
            },
            True,
        ),
        (
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 1, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            True,
        ),
        (
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            False,
        ),
        (
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [10, 2],
                    FEATURE_TYPE_SEQUENCE: [18, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [3], FEATURE_TYPE_SENTENCE: []},
            },
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            False,
        ),
    ],
)
def test_raise_exception_decreased_sparse_feature_sizes(
    new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    raise_exception: bool,
):
    """Tests if exception is raised when sparse feature sizes decrease
    during incremental training."""
    if raise_exception:
        with pytest.raises(Exception) as exec_info:
            TransformerRasaModel._check_if_sparse_feature_sizes_decreased(
                new_sparse_feature_sizes=new_sparse_feature_sizes,
                old_sparse_feature_sizes=old_sparse_feature_sizes,
            )
        assert "Sparse feature sizes have decreased" in str(exec_info.value)
    else:
        TransformerRasaModel._check_if_sparse_feature_sizes_decreased(
            new_sparse_feature_sizes=new_sparse_feature_sizes,
            old_sparse_feature_sizes=old_sparse_feature_sizes,
        )


@pytest.mark.parametrize(
    "new_sparse_feature_sizes, old_sparse_feature_sizes, expected_output",
    [
        (
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [], FEATURE_TYPE_SENTENCE: [5]},
            },
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [], FEATURE_TYPE_SENTENCE: [2]},
            },
            True,
        ),
        (
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 10],
                    FEATURE_TYPE_SEQUENCE: [3, 10, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            True,
        ),
        (
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            {
                TEXT: {
                    FEATURE_TYPE_SENTENCE: [5, 2],
                    FEATURE_TYPE_SEQUENCE: [3, 5, 10],
                },
                LABEL: {FEATURE_TYPE_SEQUENCE: [2], FEATURE_TYPE_SENTENCE: []},
            },
            False,
        ),
    ],
)
def test_if_sparse_feature_sizes_have_increased(
    new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    expected_output: bool,
):
    """Tests if any of the sparse feature sizes has increased."""
    output = TransformerRasaModel._sparse_feature_sizes_have_increased(
        new_sparse_feature_sizes=new_sparse_feature_sizes,
        old_sparse_feature_sizes=old_sparse_feature_sizes,
    )
    assert output == expected_output
