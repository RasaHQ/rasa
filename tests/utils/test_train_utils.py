import pytest
import scipy.sparse
import numpy as np

from rasa.utils.train_utils import (
    SessionDataType,
    shuffle_session_data,
    split_session_data_by_label_ids,
    train_val_split,
    session_data_for_ids,
    get_number_of_examples,
    gen_batch,
    balance_session_data,
)


@pytest.fixture
async def model_data() -> SessionDataType:
    return {
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
                    np.array([0, 1, 1, 0, 2]),
                    np.array([2, 0]),
                    np.array([0, 1, 1]),
                    np.array([0, 1]),
                    np.array([0, 0, 0]),
                ]
            )
        ],
    }


def test_shuffle_session_data(model_data):
    shuffeled_session_data = shuffle_session_data(model_data)

    assert np.array(shuffeled_session_data.values()) != np.array(model_data.values())


def test_split_session_data_by_label(model_data):
    split_session_data = split_session_data_by_label_ids(
        model_data, model_data["intent_ids"][0], np.array([0, 1])
    )

    assert len(split_session_data) == 2
    for s in split_session_data:
        assert len(set(s["intent_ids"][0])) == 1


def test_train_val_split(model_data):
    train_session_data, val_session_data = train_val_split(
        model_data, 2, 42, "intent_ids"
    )

    for k, values in model_data.items():
        assert len(values) == len(train_session_data[k])
        assert len(values) == len(val_session_data[k])
        for i, v in enumerate(values):
            assert v[0].dtype == train_session_data[k][i][0].dtype

    for values in train_session_data.values():
        for v in values:
            assert v.shape[0] == 3

    for values in val_session_data.values():
        for v in values:
            assert v.shape[0] == 2


@pytest.mark.parametrize("size", [0, 1, 5])
def test_train_val_split_incorrect_size(model_data, size):
    with pytest.raises(ValueError):
        train_val_split(model_data, size, 42, "intent_ids")


def test_session_data_for_ids(model_data):
    filtered_session_data = session_data_for_ids(model_data, np.array([0, 1]))

    for values in filtered_session_data.values():
        for v in values:
            assert v.shape[0] == 2

    k = list(model_data.keys())[0]

    assert np.all(
        np.array(filtered_session_data[k][0][0]) == np.array(model_data[k][0][0])
    )
    assert np.all(
        np.array(filtered_session_data[k][0][1]) == np.array(model_data[k][0][1])
    )


def test_get_number_of_examples(model_data):
    num = get_number_of_examples(model_data)

    assert num == 5


def test_get_number_of_examples_raises_value_error(model_data):
    model_data["dense"] = np.random.randint(5, size=(2, 10))
    with pytest.raises(ValueError):
        get_number_of_examples(model_data)


def test_gen_batch(model_data):
    iterator = gen_batch(
        model_data, 2, "intent_ids", shuffle=True, batch_strategy="balanced"
    )

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


def test_balance_session_data(model_data):
    balanced_session_data = balance_session_data(model_data, 2, False, "intent_ids")

    for k, values in model_data.items():
        assert k in balanced_session_data

        for i, v in enumerate(values):
            assert len(v) == len(balanced_session_data[k][i])

    assert np.all(balanced_session_data["intent_ids"][0] == np.array([0, 1, 1, 0, 1]))
