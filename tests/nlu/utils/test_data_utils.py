import pytest
import numpy as np
from typing import List, Optional, Text
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.utils.data_utils import remove_unfeaturized_messages


@pytest.mark.parametrize(
    "messages, attribute, featurizers, should_remove",
    [
        (
            [
                Message(
                    data={TEXT: "some text"},
                    features=[
                        Features(
                            features=np.zeros(1),
                            feature_type=FEATURE_TYPE_SEQUENCE,
                            attribute=TEXT,
                            origin="feat1",
                        ),
                    ],
                ),
            ],
            TEXT,
            ["feat1"],
            False,
        ),
        (
            [
                Message(
                    data={TEXT: "some text"},
                    features=[
                        Features(
                            features=np.zeros(1),
                            feature_type=FEATURE_TYPE_SENTENCE,
                            attribute=TEXT,
                            origin="feat1",
                        ),
                    ],
                ),
            ],
            TEXT,
            ["feat2"],
            True,
        ),
        (
            [
                Message(
                    data={TEXT: "some text"},
                ),
            ],
            TEXT,
            None,
            True,
        ),
        (
            [
                Message(
                    data={TEXT: "some text"},
                    features=[
                        Features(
                            features=np.zeros(1),
                            feature_type=FEATURE_TYPE_SEQUENCE,
                            attribute=TEXT,
                            origin="feat1",
                        ),
                    ],
                ),
            ],
            TEXT,
            None,
            False,
        ),
    ],
)
def test_remove_unfeaturized_messages(
    messages: List[Message],
    attribute: Text,
    featurizers: Optional[List[Text]],
    should_remove: bool,
):
    filtered_messages = remove_unfeaturized_messages(
        messages=messages, attribute=attribute, featurizers=featurizers
    )
    if should_remove:
        assert not filtered_messages
    else:
        assert filtered_messages == messages
