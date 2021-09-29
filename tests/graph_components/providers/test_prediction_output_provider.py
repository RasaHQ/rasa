from typing import Text, Tuple

import pytest

from rasa.core.policies.policy import PolicyPrediction
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.prediction_output_provider import (
    PredictionOutputProvider,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message


@pytest.mark.parametrize(
    "inputs, output",
    [
        ((), (),),
        (
            ("parsed_messages", "tracker_with_added_message", "ensemble_output",),
            ("parsed_message", "ensemble_tracker", "prediction",),
        ),
        (
            ("parsed_messages", "tracker_with_added_message",),
            ("parsed_message", "tracker",),
        ),
        (
            ("parsed_messages", "ensemble_output",),
            ("parsed_message", "ensemble_tracker", "prediction",),
        ),
        (("ensemble_output",), ("ensemble_tracker", "prediction",),),
        (("parsed_messages",), ("parsed_message",)),
    ],
)
def test_prediction_output_providor_provides_outputs(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    inputs: Tuple[Text],
    output: Tuple[Text],
):
    component = PredictionOutputProvider.create(
        PredictionOutputProvider.get_default_config(),
        default_model_storage,
        Resource(""),
        default_execution_context,
    )

    input_values = {
        "parsed_messages": [Message(Text="Some message")],
        "tracker_with_added_message": DialogueStateTracker("tracker", []),
        "ensemble_output": (
            DialogueStateTracker("ensemble_tracker", []),
            PolicyPrediction([1, 0], "policy"),
        ),
    }
    kwargs = {}
    for input_name in inputs:
        kwargs[input_name] = input_values[input_name]

    expected_output = []
    if "parsed_message" in output:
        expected_output.append(input_values["parsed_messages"][0])
    else:
        expected_output.append(None)

    if "ensemble_tracker" in output:
        expected_output.append(input_values["ensemble_output"][0])
    elif "tracker" in output:
        expected_output.append(input_values["tracker_with_added_message"])
    else:
        expected_output.append(None)

    if "prediction" in output:
        expected_output.append(input_values["ensemble_output"][1])
    else:
        expected_output.append(None)

    result = component.provide(**kwargs)

    assert result == tuple(expected_output)
