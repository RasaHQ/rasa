import pytest
import numpy as np

from rasa.utils.train_utils import (
    SessionData,
    shuffle_session_data,
    split_session_data_by_label,
    train_val_split,
    session_data_for_ids,
)


@pytest.fixture
async def session_data() -> SessionData:
    return SessionData(
        X={
            "sparse": np.random.rand(5, 10),
            "dense": np.random.randint(5, size=(5, 10)),
        },
        Y={"Y": np.random.randint(2, size=(5, 10))},
        labels={
            "tags": np.random.randint(2, size=(5, 10)),
            "labels": np.random.randint(2, size=(5)),
        },
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
        session_data, "labels", np.array([1, 2, 3, 4, 5])
    )

    assert len(split_session_data) == 5
    for s in split_session_data:
        assert len(set(s.labels["labels"])) <= 1


def test_train_val_split(session_data: SessionData):
    train_session_data, val_session_data = train_val_split(
        session_data, 2, 42, "labels"
    )

    for v in train_session_data.X.values():
        assert len(v) == 3

    for v in val_session_data.X.values():
        assert len(v) == 2


def test_session_data_for_ids(session_data: SessionData):
    filtered_session_data = session_data_for_ids(session_data, np.array([0, 1]))

    for v in filtered_session_data.X.values():
        assert len(v) == 2

    k = list(session_data.X.keys())[0]

    assert np.all(
        np.array(filtered_session_data.X[k][0]) == np.array(session_data.X[k][0])
    )
    assert np.all(
        np.array(filtered_session_data.X[k][1]) == np.array(session_data.X[k][1])
    )
