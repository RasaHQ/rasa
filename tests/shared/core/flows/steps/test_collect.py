from typing import Any, Dict, Optional

import pytest

from rasa.shared.core.flows.flow_step_links import (
    FlowStepLinks,
    StaticFlowStepLink,
)
from rasa.shared.core.flows.steps import CollectInformationFlowStep
from rasa.shared.core.flows.steps.collect import (
    DTMFConfig,
    PerChannelSilenceTimeout,
    SilenceTimeout,
    SingleSilenceTimeout,
)
from rasa.shared.core.slots import SlotRejection
from rasa.shared.exceptions import RasaException


@pytest.mark.parametrize(
    "data, expected_dtmf_config",
    [
        (
            {"length": 6},
            DTMFConfig(length=6, finish_on_key=None, allow_audio_input=True),
        ),
        (
            {"finish_on_key": "#"},
            DTMFConfig(length=None, finish_on_key="#", allow_audio_input=True),
        ),
        (
            {"length": 4, "allow_audio_input": False},
            DTMFConfig(length=4, finish_on_key=None, allow_audio_input=False),
        ),
        (
            {"finish_on_key": "*", "allow_audio_input": False},
            DTMFConfig(length=None, finish_on_key="*", allow_audio_input=False),
        ),
        (
            {"length": 10, "allow_audio_input": True},
            DTMFConfig(length=10, finish_on_key=None, allow_audio_input=True),
        ),
    ],
)
def test_dtmf_config_from_json(
    data: Dict[str, Any], expected_dtmf_config: DTMFConfig
) -> None:
    """Test that DTMFConfig can be created from JSON with valid inputs."""
    dtmf_config = DTMFConfig.from_json(data)

    assert dtmf_config.length == expected_dtmf_config.length
    assert dtmf_config.finish_on_key == expected_dtmf_config.finish_on_key
    assert dtmf_config.allow_audio_input == expected_dtmf_config.allow_audio_input


def test_dtmf_config_from_json_both_length_and_finish_on_key() -> None:
    """DTMFConfig raises exception when both length and finish_on_key are set."""
    data = {"length": 6, "finish_on_key": "#"}

    with pytest.raises(RasaException) as exc_info:
        DTMFConfig.from_json(data)

    assert "cannot have both 'length' and 'finish_on_key'" in str(exc_info.value)


@pytest.mark.parametrize(
    "data, expected_silence_timeout",
    [
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "ask_before_filling": True,
                "reset_after_flow_ends": False,
                "rejections": [{"if": "condition", "utter": "sample_a"}],
                "force_slot_filling": True,
                "silence_timeout": 10.0,
            },
            SingleSilenceTimeout(10.0),
        ),
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "ask_before_filling": True,
                "reset_after_flow_ends": False,
                "rejections": [{"if": "condition", "utter": "sample_a"}],
                "force_slot_filling": True,
                "silence_timeout": {
                    "channel_a": 5.0,
                    "channel_b": 15.0,
                },
            },
            PerChannelSilenceTimeout(
                {
                    "channel_a": 5.0,
                    "channel_b": 15.0,
                }
            ),
        ),
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "ask_before_filling": True,
                "reset_after_flow_ends": False,
                "rejections": [{"if": "condition", "utter": "sample_a"}],
                "force_slot_filling": True,
            },
            None,
        ),
    ],
)
def test_collect_step_from_json(
    data: Dict[str, Any], expected_silence_timeout: Optional[SilenceTimeout]
) -> None:
    """Test that CollectInformationFlowStep can be created from JSON."""
    step = CollectInformationFlowStep.from_json("flow_id", data)

    assert step.collect == "test_slot"
    assert step.utter == "utter_ask_test_slot"
    assert step.ask_before_filling is True
    assert step.reset_after_flow_ends is False
    assert len(step.rejections) == 1
    assert step.rejections[0].if_ == "condition"
    assert step.rejections[0].utter == "sample_a"
    assert step.force_slot_filling is True
    assert step.silence_timeout == expected_silence_timeout


@pytest.mark.parametrize(
    "data, expected_dtmf_length, expected_dtmf_finish_key, expected_allow_audio",
    [
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "dtmf": {"length": 6},
            },
            6,
            None,
            True,
        ),
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "dtmf": {"finish_on_key": "#"},
            },
            None,
            "#",
            True,
        ),
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "dtmf": {"length": 4, "allow_audio_input": False},
            },
            4,
            None,
            False,
        ),
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "dtmf": {"finish_on_key": "*", "allow_audio_input": True},
            },
            None,
            "*",
            True,
        ),
    ],
)
def test_collect_step_from_json_with_dtmf(
    data: Dict[str, Any],
    expected_dtmf_length: Optional[int],
    expected_dtmf_finish_key: Optional[str],
    expected_allow_audio: bool,
) -> None:
    """CollectInformationFlowStep can be created from JSON with DTMF config."""
    step = CollectInformationFlowStep.from_json("flow_id", data)

    assert step.collect == "test_slot"
    assert step.utter == "utter_ask_test_slot"
    assert step.dtmf is not None
    assert step.dtmf.length == expected_dtmf_length
    assert step.dtmf.finish_on_key == expected_dtmf_finish_key
    assert step.dtmf.allow_audio_input == expected_allow_audio


def test_collect_step_from_json_with_invalid_dtmf() -> None:
    """CollectInformationFlowStep raises exception with invalid DTMF config."""
    data = {
        "collect": "test_slot",
        "utter": "utter_ask_test_slot",
        "dtmf": {"length": 6, "finish_on_key": "#"},
    }

    with pytest.raises(RasaException) as exc_info:
        CollectInformationFlowStep.from_json("flow_id", data)

    assert "cannot have both 'length' and 'finish_on_key'" in str(exc_info.value)


@pytest.mark.parametrize(
    "data, expected_exception",
    [
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "ask_before_filling": True,
                "reset_after_flow_ends": False,
                "rejections": [{"if": "condition", "utter": "sample_a"}],
                "force_slot_filling": True,
                "silence_timeout": -1,
            },
            RasaException(
                "Invalid silence timeout value: -1. "
                "Silence timeout must be a non-negative number."
            ),
        ),
        (
            {
                "collect": "test_slot",
                "utter": "utter_ask_test_slot",
                "ask_before_filling": True,
                "reset_after_flow_ends": False,
                "rejections": [{"if": "condition", "utter": "sample_a"}],
                "force_slot_filling": True,
                "silence_timeout": "a",
            },
            RasaException(
                "Invalid silence timeout value: -1. "
                "If defined at collect step, silence timeout must be a number."
            ),
        ),
    ],
)
def test_collect_step_from_json_invalid_silence_timeout(
    data: Dict[str, Any], expected_exception: RasaException
) -> None:
    """Test that CollectInformationFlowStep can be created from JSON."""
    with pytest.raises(RasaException) as exc_info:
        CollectInformationFlowStep.from_json("flow_id", data)
        assert exc_info.value == expected_exception


@pytest.mark.parametrize(
    "silence_timeout",
    [
        SingleSilenceTimeout(10.0),
        PerChannelSilenceTimeout(
            {
                "channel_a": 5.0,
                "channel_b": 15.0,
            }
        ),
    ],
)
def test_collect_step_as_json(
    silence_timeout: SilenceTimeout,
) -> None:
    """Test that CollectInformationFlowStep can be serialized to JSON."""
    step = CollectInformationFlowStep(
        flow_id="flow_id",
        collect="test_slot",
        utter="utter_ask_test_slot",
        ask_before_filling=True,
        reset_after_flow_ends=False,
        rejections=[SlotRejection(if_="condition", utter="sample_a")],
        force_slot_filling=True,
        silence_timeout=silence_timeout,
        custom_id=None,
        idx=0,
        description="Collect test slot",
        metadata={},
        next=FlowStepLinks(links=[]),
        collect_action="action_ask_test_slot",
    )

    json_data = step.as_json()

    assert json_data["collect"] == "test_slot"
    assert json_data["utter"] == "utter_ask_test_slot"
    assert json_data["ask_before_filling"] is True
    assert json_data["reset_after_flow_ends"] is False
    assert json_data["rejections"] == [{"if": "condition", "utter": "sample_a"}]
    assert json_data["force_slot_filling"] is True
    assert json_data["silence_timeout"] == silence_timeout.to_json().get(
        "silence_timeout"
    )
    assert json_data["description"] == "Collect test slot"
    assert "metadata" not in json_data
    assert "next" not in json_data


def test_collect_step_as_json_without_silence_timeout() -> None:
    """Test that CollectInformationFlowStep can be serialized to JSON.

    This test checks the behavior when silence_timeout is not set.
    """
    step = CollectInformationFlowStep(
        flow_id="flow_id",
        collect="test_slot",
        utter="utter_ask_test_slot",
        ask_before_filling=True,
        reset_after_flow_ends=False,
        rejections=[SlotRejection(if_="condition", utter="sample_a")],
        force_slot_filling=True,
        custom_id=None,
        idx=0,
        description="Collect test slot",
        metadata={},
        next=FlowStepLinks(links=[]),
        collect_action="action_ask_test_slot",
    )

    json_data = step.as_json()

    assert json_data["collect"] == "test_slot"
    assert json_data["utter"] == "utter_ask_test_slot"
    assert json_data["ask_before_filling"] is True
    assert json_data["reset_after_flow_ends"] is False
    assert json_data["rejections"] == [{"if": "condition", "utter": "sample_a"}]
    assert json_data["force_slot_filling"] is True
    assert json_data["description"] == "Collect test slot"
    assert "metadata" not in json_data
    assert "next" not in json_data
    assert "silence_timeout" not in json_data


def test_collect_step_as_json_with_next_and_metadata() -> None:
    """Test that CollectInformationFlowStep can be serialized to JSON.

    This test checks the behavior when next step and metadata are provided.
    """
    next_step = FlowStepLinks(
        links=[StaticFlowStepLink(target_step_id="some_next_step")]
    )

    step = CollectInformationFlowStep(
        flow_id="flow_id",
        collect="test_slot",
        utter="utter_ask_test_slot",
        ask_before_filling=True,
        reset_after_flow_ends=False,
        rejections=[SlotRejection(if_="condition", utter="sample_a")],
        force_slot_filling=True,
        silence_timeout=SingleSilenceTimeout(10.0),
        custom_id=None,
        idx=0,
        description="Collect test slot",
        collect_action="action_ask_test_slot",
        next=next_step,
        metadata={
            "key1": "value1",
            "key2": 42,
            "key3": True,
        },
    )

    json_data = step.as_json()

    assert json_data["collect"] == "test_slot"
    assert json_data["utter"] == "utter_ask_test_slot"
    assert json_data["ask_before_filling"] is True
    assert json_data["reset_after_flow_ends"] is False
    assert json_data["rejections"] == [{"if": "condition", "utter": "sample_a"}]
    assert json_data["force_slot_filling"] is True
    assert json_data["silence_timeout"] == 10.0
    assert json_data["description"] == "Collect test slot"
    assert json_data["next"] == next_step.as_json()
    assert json_data["metadata"] == {
        "key1": "value1",
        "key2": 42,
        "key3": True,
    }
