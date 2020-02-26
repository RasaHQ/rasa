import copy

import pytest
import scipy.sparse
import numpy as np

from rasa.utils.tensorflow.model_data import RasaModelData


@pytest.fixture
async def model_data() -> RasaModelData:
    return RasaModelData(
        label_key="intent_ids",
        data={
            "text_features": [
                np.array(
                    [
                        np.random.rand(5, 14),
                        np.random.rand(2, 14),
                        np.random.rand(3, 14),
                        np.random.rand(1, 14),
                        np.random.rand(3, 14),
                    ]
                ),
                np.array(
                    [
                        scipy.sparse.csr_matrix(np.random.randint(5, size=(5, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(5, size=(2, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(5, size=(3, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(5, size=(1, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(5, size=(3, 10))),
                    ]
                ),
            ],
            "intent_features": [
                np.array(
                    [
                        np.random.randint(2, size=(5, 10)),
                        np.random.randint(2, size=(2, 10)),
                        np.random.randint(2, size=(3, 10)),
                        np.random.randint(2, size=(1, 10)),
                        np.random.randint(2, size=(3, 10)),
                    ]
                )
            ],
            "intent_ids": [np.array([0, 1, 0, 1, 1])],
            "tag_ids": [
                np.array(
                    [
                        np.array([[0], [1], [1], [0], [2]]),
                        np.array([[2], [0]]),
                        np.array([[0], [1], [1]]),
                        np.array([[0], [1]]),
                        np.array([[0], [0], [0]]),
                    ]
                )
            ],
        },
    )


def test_shuffle_session_data(model_data: RasaModelData):
    before = copy.copy(model_data)

    # precondition
    assert np.all(
        np.array(list(before.values())) == np.array(list(model_data.values()))
    )

    data = model_data._shuffled_data(model_data.data)

    # check that original data didn't change
    assert np.all(
        np.array(list(before.values())) == np.array(list(model_data.values()))
    )
    # check that new data is different
    assert np.all(np.array(model_data.values()) != np.array(data.values()))


def test_split_data_by_label(model_data: RasaModelData):
    split_model_data = model_data._split_by_label_ids(
        model_data.data, model_data.get("intent_ids")[0], np.array([0, 1])
    )

    assert len(split_model_data) == 2
    for s in split_model_data:
        assert len(set(s.get("intent_ids")[0])) == 1


def test_split_data_by_none_label(model_data: RasaModelData):
    model_data.label_key = None

    split_model_data = model_data.split(2, 42)

    assert len(split_model_data) == 2

    train_data = split_model_data[0]
    test_data = split_model_data[1]

    # train data should have 3 examples
    assert len(train_data.get("intent_ids")[0]) == 3
    # test data should have 2 examples
    assert len(test_data.get("intent_ids")[0]) == 2


def test_train_val_split(model_data: RasaModelData):
    train_model_data, test_model_data = model_data.split(2, 42)

    for k, values in model_data.items():
        assert len(values) == len(train_model_data.get(k))
        assert len(values) == len(test_model_data.get(k))
        for i, v in enumerate(values):
            assert v[0].dtype == train_model_data.get(k)[i][0].dtype

    for values in train_model_data.values():
        for v in values:
            assert v.shape[0] == 3

    for values in test_model_data.values():
        for v in values:
            assert v.shape[0] == 2


@pytest.mark.parametrize("size", [0, 1, 5])
def test_train_val_split_incorrect_size(model_data: RasaModelData, size: int):
    with pytest.raises(ValueError):
        model_data.split(size, 42)


def test_session_data_for_ids(model_data: RasaModelData):
    filtered_data = model_data._data_for_ids(model_data.data, np.array([0, 1]))

    for values in filtered_data.values():
        for v in values:
            assert v.shape[0] == 2

    k = list(model_data.keys())[0]

    assert np.all(np.array(filtered_data[k][0][0]) == np.array(model_data.get(k)[0][0]))
    assert np.all(np.array(filtered_data[k][0][1]) == np.array(model_data.get(k)[0][1]))


def test_get_number_of_examples(model_data: RasaModelData):
    assert model_data.number_of_examples() == 5


def test_get_number_of_examples_raises_value_error(model_data: RasaModelData):
    model_data.data["dense"] = [np.random.randint(5, size=(2, 10))]
    with pytest.raises(ValueError):
        model_data.number_of_examples()


def test_gen_batch(model_data: RasaModelData):
    iterator = model_data._gen_batch(2, shuffle=True, batch_strategy="balanced")
    print(model_data.data["tag_ids"][0])
    batch = next(iterator)
    assert len(batch) == 7
    assert len(batch[0]) == 2

    batch = next(iterator)
    assert len(batch) == 7
    assert len(batch[0]) == 2

    batch = next(iterator)
    assert len(batch) == 7
    assert len(batch[0]) == 1

    with pytest.raises(StopIteration):
        next(iterator)


def test_balance_model_data(model_data: RasaModelData):
    data = model_data._balanced_data(model_data.data, 2, False)

    assert np.all(data.get("intent_ids")[0] == np.array([0, 1, 1, 0, 1]))


def test_not_balance_model_data(model_data: RasaModelData):
    test_model_data = RasaModelData(label_key="tag_ids", data=model_data.data)

    data = test_model_data._balanced_data(test_model_data.data, 2, False)

    assert np.all(data.get("tag_ids") == test_model_data.get("tag_ids"))


def test_get_num_of_features(model_data: RasaModelData):
    num_features = model_data.feature_dimension("text_features")

    assert num_features == 24
