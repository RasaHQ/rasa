from unittest.mock import MagicMock

import pytest

from rasa.dialogue_understanding.generator import LLMBasedCommandGenerator
from rasa.llm_fine_tuning.utils import (
    make_mock_invoke_llm,
    patch_invoke_llm_in_generators,
)
from rasa.shared.providers.llm.llm_response import LLMResponse


@pytest.mark.asyncio
async def test_make_mock_invoke_llm_returns_expected_response():
    wanted_commands = "MOVE_FORWARD"
    mock_invoke = make_mock_invoke_llm(wanted_commands)

    dummy_self = MagicMock(spec=LLMBasedCommandGenerator)

    response: LLMResponse = await mock_invoke(dummy_self, "ignored prompt")

    assert response.choices == [wanted_commands]
    assert isinstance(response, LLMResponse)
    # just a couple of sanity checks on the synthetic meta-data
    assert response.created > 0
    assert response.model == ("mocked-llm")


@pytest.mark.asyncio
async def test_patch_replaces_and_restores_everywhere() -> None:
    """
    patch_invoke_llm_in_generators must
      • replace invoke_llm on the base class and on every subclass that exists
        when the context manager is entered;
      • restore the previous attributes when the context manager exits.
    """

    # Build an *abstract* hierarchy that the helper must walk through.
    class Sub1(LLMBasedCommandGenerator):
        pass

    class Sub2(Sub1):
        pass

    # Keep references to the 'old' attributes so we can compare later.
    original_base_impl = LLMBasedCommandGenerator.invoke_llm
    original_sub1_impl = Sub1.invoke_llm
    original_sub2_impl = Sub2.invoke_llm

    # The mock implementation we want the context manager to install.
    wanted_commands = "start flow test"
    mock_impl = make_mock_invoke_llm(wanted_commands)

    # Inside the context manager everything must be patched …
    with patch_invoke_llm_in_generators(mock_impl):
        assert LLMBasedCommandGenerator.invoke_llm is mock_impl
        assert Sub1.invoke_llm is mock_impl
        assert Sub2.invoke_llm is mock_impl

        # Call the method once through the patch:
        fake_self = MagicMock(spec=LLMBasedCommandGenerator)
        response: LLMResponse = await Sub2.invoke_llm(fake_self, "ignored")
        assert response.choices == [wanted_commands]

    # everything must be back to normal after exiting the context manager.
    assert LLMBasedCommandGenerator.invoke_llm is original_base_impl
    assert Sub1.invoke_llm is original_sub1_impl
    assert Sub2.invoke_llm is original_sub2_impl


def test_patch_restores_even_on_exception() -> None:
    """
    If user code raises while the patch is active, the helper still has to
    restore the original methods.  Use a meaningful command string and a
    clear exception message.
    """

    class TestLLMCommandGenerator(LLMBasedCommandGenerator):
        pass

    original_impl = TestLLMCommandGenerator.invoke_llm
    mock_impl = make_mock_invoke_llm("REPORT_STATUS")

    with pytest.raises(RuntimeError, match="simulated failure inside with-block"):
        with patch_invoke_llm_in_generators(mock_impl):
            assert TestLLMCommandGenerator.invoke_llm is mock_impl
            # Raise to trigger the finally-branch of the context manager
            raise RuntimeError("simulated failure inside with-block")

    # Original `invoke_llm` must be restored even if the patch failed.
    assert TestLLMCommandGenerator.invoke_llm is original_impl
