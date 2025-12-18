import numpy as np

from rasa.core.turns.state.state_featurizers import BasicStateFeaturizer
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import ACTION_NAME


def test_convert_action_labels_to_ids(domain: Domain):
    trackers_as_actions = [
        ["utter_greet", "utter_channel"],
        ["utter_greet", "utter_default", "utter_goodbye"],
    ]

    turn_featurizer = BasicStateFeaturizer()
    turn_featurizer.train(domain=domain)

    # TODO: replace this
    actual_output = turn_featurizer._encoders[
        ACTION_NAME
    ].encode_as_array_of_index_arrays(trackers_as_actions)

    expected_output = np.array(
        [
            np.array(
                [
                    domain.action_names_or_texts.index("utter_greet"),
                    domain.action_names_or_texts.index("utter_channel"),
                ],
            ),
            np.array(
                [
                    domain.action_names_or_texts.index("utter_greet"),
                    domain.action_names_or_texts.index("utter_default"),
                    domain.action_names_or_texts.index("utter_goodbye"),
                ],
            ),
        ],
    )

    assert expected_output.size == actual_output.size
    for expected_array, actual_array in zip(expected_output, actual_output):
        assert np.all(expected_array == actual_array)


"""
def test_convert_intent_labels_to_ids(domain: Domain):
    trackers_as_intents = [
        ["next_intent", "nlu_fallback", "out_of_scope", "restart"],
        ["greet", "hello", "affirm"],
    ]

    turn_featurizer = StatefulTurnFeaturizer()
    turn_featurizer.train(domain=domain)

    # TODO: replace this
    actual_labels = turn_featurizer._multihot_encoders[ACTION_NAME].encode_as_index_matrix(trackers_as_intents)

    expected_labels = np.array(
        [
            [
                domain.intents.index("next_intent"),
                domain.intents.index("nlu_fallback"),
                domain.intents.index("out_of_scope"),
                domain.intents.index("restart"),
            ],
            [
                domain.intents.index("greet"),
                domain.intents.index("hello"),
                domain.intents.index("affirm"),

                # FIXME: is this actually used?
                LABEL_PAD_ID,
            ],
        ],
    )
    assert expected_labels.size == actual_labels.size
    assert expected_labels.shape == actual_labels.shape
    assert np.all(expected_labels == actual_labels)
"""
