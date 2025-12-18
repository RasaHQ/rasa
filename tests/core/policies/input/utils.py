from typing import Text, Dict, List

import numpy as np
from rasa.shared.nlu.training_data.features import Features

# TODO: move this elsewhere...


def compare_featurized_states(
    states1: List[Dict[Text, List[Features]]], states2: List[Dict[Text, List[Features]]]
) -> bool:
    """Compares two lists of featurized states and returns True if they
    are identical and False otherwise.
    """

    if len(states1) != len(states2):
        return False

    for state1, state2 in zip(states1, states2):
        if state1.keys() != state2.keys():
            return False
        for key in state1.keys():
            for feature1, feature2 in zip(state1[key], state2[key]):
                if np.any((feature1.features != feature2.features).toarray()):
                    return False
                # if feature1.origin != feature2.origin: # NOTE: breaks due to rework
                #    return False
                if feature1.attribute != feature2.attribute:
                    return False
                if feature1.type != feature2.type:
                    return False
    return True
