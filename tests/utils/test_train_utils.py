import pytest
import scipy.sparse
import numpy as np

from rasa.utils.train_utils import (
    SessionData,
    shuffle_session_data,
    split_session_data_by_label,
    train_val_split,
    session_data_for_ids,
    get_number_of_examples,
    gen_batch,
    balance_session_data,
)


@pytest.fixture
async def session_data() -> SessionData:
    return SessionData(
        X={
            "dense": np.random.rand(5, 10),
            "sparse": scipy.sparse.csr_matrix(np.random.randint(5, size=(5, 10))),
        },
        Y={"Y": np.random.randint(2, size=(5, 10))},
        labels={"labels": np.array([0, 1, 0, 0, 0])},
    )


def test_shuffle_session_data(session_data: SessionData):
    shuffeled_session_data = shuffle_session_data(session_data)

    assert np.array(shuffeled_session_data.X.values()) != np.array(
        session_data.X.values()
    )
    assert np.array(shuffeled_session_data.Y.values()) != np.array(
        session_data.Y.values()
    )
    assert np.array(shuffeled_session_data.labels.values()) != np.array(
        session_data.labels.values()
    )


def test_split_session_data_by_label(session_data: SessionData):
    split_session_data = split_session_data_by_label(
        session_data, "labels", np.array([0, 1])
    )

    assert len(split_session_data) == 2
    for s in split_session_data:
        assert len(set(s.labels["labels"])) == 1


def test_split_session_data_by_incorrect_label(session_data: SessionData):
    with pytest.raises(ValueError):
        split_session_data_by_label(
            session_data, "not-existing", np.array([1, 2, 3, 4, 5])
        )


def test_train_val_split(session_data: SessionData):
    train_session_data, val_session_data = train_val_split(
        session_data, 2, 42, "labels"
    )

    for v in train_session_data.X.values():
        assert v.shape[0] == 3

    for v in val_session_data.X.values():
        assert v.shape[0] == 2


@pytest.mark.parametrize("size", [0, 1, 5])
def test_train_val_split_incorrect_size(session_data: SessionData, size):
    with pytest.raises(ValueError):
        train_val_split(session_data, size, 42, "labels")


def test_session_data_for_ids(session_data: SessionData):
    filtered_session_data = session_data_for_ids(session_data, np.array([0, 1]))

    for v in filtered_session_data.X.values():
        assert v.shape[0] == 2

    k = list(session_data.X.keys())[0]

    assert np.all(
        np.array(filtered_session_data.X[k][0]) == np.array(session_data.X[k][0])
    )
    assert np.all(
        np.array(filtered_session_data.X[k][1]) == np.array(session_data.X[k][1])
    )


def test_get_number_of_examples(session_data: SessionData):
    num = get_number_of_examples(session_data)

    assert num == 5


def test_get_number_of_examples_raises_value_error(session_data: SessionData):
    session_data.X["dense"] = np.random.randint(5, size=(2, 10))
    with pytest.raises(ValueError):
        get_number_of_examples(session_data)


def test_gen_batch(session_data: SessionData):
    iterator = gen_batch(session_data, 2, "labels", shuffle=True)

    batch = next(iterator)
    assert len(batch) == 4
    assert len(batch[0]) == 2

    batch = next(iterator)
    assert len(batch) == 4
    assert len(batch[0]) == 2

    batch = next(iterator)
    assert len(batch) == 4
    assert len(batch[0]) == 1

    with pytest.raises(StopIteration):
        next(iterator)


def test_balance_session_data(session_data: SessionData):
    session_data.labels["labels"] = np.array([0, 0, 0, 1, 1])

    balanced_session_data = balance_session_data(session_data, 2, False, "labels")

    labels = balanced_session_data.labels["labels"]

    assert 5 == len(labels)
    assert np.all(np.array([0, 0, 1, 0, 1]) == labels)
