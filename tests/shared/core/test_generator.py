import rasa.shared.core.generator


def test_subsample_array_read_only():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = rasa.shared.core.generator._subsample_array(
        t, 5, can_modify_incoming_array=False
    )

    assert len(r) == 5
    assert set(r).issubset(t)


def test_subsample_array():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # this will modify the original array and shuffle it
    r = rasa.shared.core.generator._subsample_array(t, 5)

    assert len(r) == 5
    assert set(r).issubset(t)
