from rasa.core.policies.flows.flow_exceptions import (
    FlowCircuitBreakerTrippedException,
    FlowException,
    NoNextStepInFlowException,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.exceptions import RasaException


def test_flow_circuit_breaker_tripped_exception_is_rasa_exception():
    # important, because we treat internal exceptions differently
    stack = DialogueStack(frames=[])
    e = FlowCircuitBreakerTrippedException(stack, 42)
    assert isinstance(e, RasaException)


def test_no_next_step_in_flow_exception_is_rasa_exception():
    # important, because we treat internal exceptions differently
    stack = DialogueStack(frames=[])
    e = NoNextStepInFlowException(stack)
    assert isinstance(e, RasaException)


def test_flow_exception_is_rasa_exception():
    # important, because we treat internal exceptions differently
    e = FlowException()
    assert isinstance(e, RasaException)
