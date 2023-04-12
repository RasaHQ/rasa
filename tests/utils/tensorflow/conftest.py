import pytest
import scipy.sparse
import numpy as np

from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureArray,
    ragged_array_to_ndarray,
)


@pytest.fixture
async def model_data() -> RasaModelData:
    return RasaModelData(
        label_key="label",
        label_sub_key="ids",
        data={
            "text": {
                "sentence": [
                    FeatureArray(
                        ragged_array_to_ndarray(
                            [
                                np.random.rand(5, 14),
                                np.random.rand(2, 14),
                                np.random.rand(3, 14),
                                np.random.rand(1, 14),
                                np.random.rand(3, 14),
                            ]
                        ),
                        number_of_dimensions=3,
                    ),
                    FeatureArray(
                        ragged_array_to_ndarray(
                            [
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(5, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(2, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(3, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(1, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(3, 10))
                                ),
                            ]
                        ),
                        number_of_dimensions=3,
                    ),
                ]
            },
            "action_text": {
                "sequence": [
                    FeatureArray(
                        ragged_array_to_ndarray(
                            [
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(5, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(2, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(1, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    ),
                                ],
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(5, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(2, 10))
                                    ),
                                ],
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(5, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(1, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    ),
                                ],
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    )
                                ],
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(1, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(7, 10))
                                    ),
                                ],
                            ]
                        ),
                        number_of_dimensions=4,
                    ),
                    FeatureArray(
                        ragged_array_to_ndarray(
                            [
                                [
                                    np.random.rand(5, 14),
                                    np.random.rand(2, 14),
                                    np.random.rand(3, 14),
                                    np.random.rand(1, 14),
                                    np.random.rand(3, 14),
                                ],
                                [np.random.rand(5, 14), np.random.rand(2, 14)],
                                [
                                    np.random.rand(5, 14),
                                    np.random.rand(1, 14),
                                    np.random.rand(3, 14),
                                ],
                                [np.random.rand(3, 14)],
                                [
                                    np.random.rand(3, 14),
                                    np.random.rand(1, 14),
                                    np.random.rand(7, 14),
                                ],
                            ]
                        ),
                        number_of_dimensions=4,
                    ),
                ]
            },
            "dialogue": {
                "sentence": [
                    FeatureArray(
                        ragged_array_to_ndarray(
                            [
                                np.random.randint(2, size=(5, 10)),
                                np.random.randint(2, size=(2, 10)),
                                np.random.randint(2, size=(3, 10)),
                                np.random.randint(2, size=(1, 10)),
                                np.random.randint(2, size=(3, 10)),
                            ]
                        ),
                        number_of_dimensions=3,
                    )
                ]
            },
            "label": {
                "ids": [FeatureArray(np.array([0, 1, 0, 1, 1]), number_of_dimensions=1)]
            },
            "entities": {
                "tag_ids": [
                    FeatureArray(
                        ragged_array_to_ndarray(
                            [
                                np.array([[0], [1], [1], [0], [2]]),
                                np.array([[2], [0]]),
                                np.array([[0], [1], [1]]),
                                np.array([[0], [1]]),
                                np.array([[0], [0], [0]]),
                            ]
                        ),
                        number_of_dimensions=3,
                    )
                ]
            },
        },
    )
