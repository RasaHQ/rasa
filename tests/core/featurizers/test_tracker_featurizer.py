from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer


def test_fail_to_load_non_existent_featurizer():
    assert TrackerFeaturizer.load("non_existent_class") is None
