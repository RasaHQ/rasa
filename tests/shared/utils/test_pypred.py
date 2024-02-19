import pytest
import uuid
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils import pypred


@pytest.mark.parametrize(
    "predicate, expected",
    [
        ("slots.bar > 10", "slots.bar > 10"),
        (
            "slots.bar > 10 and slots.foo == 'bar'",
            "slots.bar > 10 and slots.foo == 'bar'",
        ),
        ("slots.user_type == 'Premium'", "slots.user_type == 'premium'"),
        ("slots.user_type == 'Premium Temp'", "slots.user_type == 'Premium Temp'"),
        (
            "slots.user_type == 'Standard' and slots.user_type == 'STANDARD'",
            "slots.user_type == 'standard' and slots.user_type == 'standard'",
        ),
        (
            "slots.user_type == 'STANDARD' and slots.user_type != 'Premium'",
            "slots.user_type == 'standard' and slots.user_type != 'premium'",
        ),
        (
            "slots.user_type == 'Standard' and slots.foo != 'BAR'",
            "slots.user_type == 'standard' and slots.foo != 'BAR'",
        ),
        (
            "slots.user_type == 'Standard' and slots.STANDARD == 'hello'",
            "slots.user_type == 'standard' and slots.STANDARD == 'hello'",
        ),
        (
            "slots.user_type == 'STANDARD-User' and slots.user_type == 'STANDARD'",
            "slots.user_type == 'STANDARD-User' and slots.user_type == 'standard'",
        ),
    ],
)
def test_pypred_get_case_insensitive_predicate(
    predicate: str,
    expected: bool,
):
    test_domain = Domain.from_yaml(
        """
        slots:
            bar:
              type: float
              initial_value: 0.0
            foo:
              type: text
              initial_value: bar
            user_type:
              type: categorical
              values:
                - Premium
                - Standard
            STANDARD:
              type: text
              initial_value: dummy
        """
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[],
        slots=test_domain.slots,
    )

    assert (
        pypred.get_case_insensitive_predicate(
            predicate, ["bar", "foo", "user_type", "STANDARD"], tracker
        )
        == expected
    )


@pytest.mark.parametrize(
    "predicate, expected",
    [
        ("slots.bar > 10", "slots.bar > 10"),
        (
            "slots.bar > 10 and slots.foo == 'bar'",
            "slots.bar > 10 and slots.foo == 'bar'",
        ),
        ("slots.user_type == 'Premium'", "slots.user_type == 'premium'"),
        ("slots.user_type == 'Premium Temp'", "slots.user_type == 'Premium Temp'"),
        (
            "slots.user_type == 'Standard' and slots.user_type == 'STANDARD'",
            "slots.user_type == 'standard' and slots.user_type == 'standard'",
        ),
        (
            "slots.user_type == 'STANDARD' and slots.user_type != 'Premium'",
            "slots.user_type == 'standard' and slots.user_type != 'premium'",
        ),
        (
            "slots.user_type == 'Standard' and slots.foo != 'BAR'",
            "slots.user_type == 'standard' and slots.foo != 'BAR'",
        ),
        (
            "slots.user_type == 'Standard' and slots.STANDARD == 'hello'",
            "slots.user_type == 'standard' and slots.STANDARD == 'hello'",
        ),
        (
            "slots.user_type == 'STANDARD-User' and slots.user_type == 'STANDARD'",
            "slots.user_type == 'STANDARD-User' and slots.user_type == 'standard'",
        ),
    ],
)
def test_pypred_get_case_insensitive_predicate_given_slot_instance(
    predicate: str,
    expected: bool,
):
    test_domain = Domain.from_yaml(
        """
        slots:
            bar:
              type: float
              initial_value: 0.0
            foo:
              type: text
              initial_value: bar
            user_type:
              type: categorical
              values:
                - Premium
                - Standard
            STANDARD:
              type: text
              initial_value: dummy
        """
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[],
        slots=test_domain.slots,
    )

    assert (
        pypred.get_case_insensitive_predicate_given_slot_instance(
            predicate, tracker.slots
        )
        == expected
    )
