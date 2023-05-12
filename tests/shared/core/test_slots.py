from typing import Any, Tuple, List, Dict, Text

import pytest
from _pytest.fixtures import SubRequest

import rasa.shared.core.constants
from rasa.shared.core.events import SlotSet
from rasa.shared.core.slots import (
    InvalidSlotTypeException,
    Slot,
    TextSlot,
    BooleanSlot,
    FloatSlot,
    ListSlot,
    CategoricalSlot,
    bool_from_any,
    AnySlot,
    InvalidSlotConfigError,
)
from rasa.shared.core.trackers import DialogueStateTracker


class SlotTestCollection:
    """Tests every slot needs to fulfill.

    Each slot can declare further tests on its own."""

    def create_slot(
        self, mappings: List[Dict[Text, Any]], influence_conversation: bool
    ) -> Slot:
        raise NotImplementedError

    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        """Values where featurization is defined and should be tested."""
        raise NotImplementedError

    def invalid_value(self, request: SubRequest) -> Any:
        """Values, that should be handled gracefully but where the
        featurization is not defined."""
        raise NotImplementedError

    @pytest.fixture()
    def mappings(self) -> List[Dict[Text, Any]]:
        return [{}]

    def test_featurization(
        self,
        value_feature_pair: Tuple[Any, List[float]],
        mappings: List[Dict[Text, Any]],
    ):
        slot = self.create_slot(mappings=mappings, influence_conversation=True)
        value, expected = value_feature_pair
        slot.value = value
        assert slot.as_feature() == expected
        assert (
            len(slot.as_feature()) == slot.feature_dimensionality()
        ), "Wrong feature dimensionality"

        # now reset the slot to get initial value again
        slot.reset()
        assert (
            slot.value == slot.initial_value
        ), "Slot should be reset to its initial value"

    def test_empty_slot_featurization(self, mappings: List[Dict[Text, Any]]):
        slot = self.create_slot(mappings=mappings, influence_conversation=True)
        assert (
            slot.value == slot.initial_value
        ), "An empty slot should be set to the initial value"
        assert len(slot.as_feature()) == slot.feature_dimensionality()

    def test_featurization_if_marked_as_unfeaturized(
        self,
        value_feature_pair: Tuple[Any, List[float]],
        mappings: List[Dict[Text, Any]],
    ):
        slot = self.create_slot(mappings=mappings, influence_conversation=False)
        value, _ = value_feature_pair
        slot.value = value

        features = slot.as_feature()
        assert features == []

        dimensions = slot.feature_dimensionality()
        assert dimensions == 0

    def test_has_a_type_name(self, mappings: List[Dict[Text, Any]]):
        slot = self.create_slot(mappings=mappings, influence_conversation=True)
        assert slot.type_name is not None
        assert type(slot) == Slot.resolve_by_type(slot.type_name)

    def test_handles_invalid_values(
        self, invalid_value: Any, mappings: List[Dict[Text, Any]]
    ):
        slot = self.create_slot(mappings=mappings, influence_conversation=True)
        slot.value = invalid_value
        assert slot.as_feature() is not None
        assert len(slot.as_feature()) == slot.feature_dimensionality()

    @pytest.mark.parametrize("influence_conversation", [True, False])
    def test_serialization(
        self, influence_conversation: bool, mappings: List[Dict[Text, Any]]
    ):
        slot = self.create_slot(mappings, influence_conversation)

        persistence_info = slot.persistence_info()

        slot_type = Slot.resolve_by_type(persistence_info.get("type"))
        recreated = slot_type(
            slot.name, **{k: v for k, v in persistence_info.items() if k != "type"}
        )

        assert isinstance(slot, slot_type)
        assert recreated.persistence_info() == persistence_info

    @pytest.mark.parametrize("influence_conversation", [True, False])
    def test_slot_has_been_set(
        self,
        influence_conversation: bool,
        value_feature_pair: Tuple[Any, List[float]],
        mappings: List[Dict[Text, Any]],
    ):
        slot = self.create_slot(mappings, influence_conversation)
        assert not slot.has_been_set
        value, _ = value_feature_pair
        slot.value = value
        assert slot.has_been_set
        slot.reset()
        assert not slot.has_been_set

    @pytest.mark.parametrize(
        "influence_conversation, slot_mappings",
        [
            (True, []),
            (True, [{"type": "from_entity", "entity": "test"}]),
            (False, []),
            (False, [{"type": "from_entity", "entity": "test"}]),
        ],
    )
    def test_slot_fingerprint_consistency(
        self, influence_conversation: bool, slot_mappings: List[Dict[Text, Any]]
    ):
        slot1 = self.create_slot(slot_mappings, influence_conversation)
        slot2 = self.create_slot(slot_mappings, influence_conversation)
        f1 = slot1.fingerprint()
        f2 = slot2.fingerprint()
        assert f1 == f2

    @pytest.mark.parametrize("influence_conversation", [True, False])
    def test_slot_fingerprint_uniqueness(
        self, influence_conversation: bool, mappings: List[Dict[Text, Any]]
    ):
        slot = self.create_slot(mappings, influence_conversation)
        f1 = slot.fingerprint()
        slot.value = "changed"
        f2 = slot.fingerprint()
        assert f1 != f2


class TestTextSlot(SlotTestCollection):
    def create_slot(
        self, mappings: List[Dict[Text, Any]], influence_conversation: bool
    ) -> Slot:
        return TextSlot(
            "test", mappings=mappings, influence_conversation=influence_conversation
        )

    @pytest.fixture(params=[1, {"a": "b"}, 2.0, [], True])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0]),
            ("", [1]),
            ("some test string", [1]),
            ("some test string 🌴", [1]),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestBooleanSlot(SlotTestCollection):
    def create_slot(
        self, mappings: List[Dict[Text, Any]], influence_conversation: bool
    ) -> Slot:
        return BooleanSlot(
            "test", mappings=mappings, influence_conversation=influence_conversation
        )

    @pytest.fixture(params=[{"a": "b"}, [], "asd", "🌴"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0]),
            (True, [1, 1]),
            ("9", [1, 0]),
            (12, [1, 0]),
            (False, [1, 0]),
            ("0", [1, 0]),
            (0, [1, 0]),
            ("true", [1, 1]),
            ("True", [1, 1]),
            ("false", [1, 0]),
            ("False", [1, 0]),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


def test_bool_from_any_raises_value_error():
    with pytest.raises(ValueError):
        bool_from_any("abc")


def test_bool_from_any_raises_type_error():
    with pytest.raises(TypeError):
        bool_from_any(None)


class TestFloatSlot(SlotTestCollection):
    def create_slot(
        self, mappings: List[Dict[Text, Any]], influence_conversation: bool = False
    ) -> Slot:
        return FloatSlot(
            "test", mappings=mappings, influence_conversation=influence_conversation
        )

    @pytest.fixture(params=[{"a": "b"}, [], "asd", "🌴"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0]),
            (True, [1, 1]),
            (2.0, [1, 1]),
            (1.0, [1, 1]),
            (0.5, [1, 0.5]),
            (0, [1, 0]),
            (-0.5, [1, 0.0]),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestListSlot(SlotTestCollection):
    def create_slot(
        self, mappings: List[Dict[Text, Any]], influence_conversation: bool
    ) -> Slot:
        return ListSlot(
            "test", mappings=mappings, influence_conversation=influence_conversation
        )

    @pytest.fixture(params=[{"a": "b"}, 1, True, "asd", "🌴"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(params=[(None, [0]), ([], [0]), ([1], [1]), (["asd", 1, {}], [1])])
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param

    @pytest.mark.parametrize("value", ["cat", ["cat"]])
    def test_apply_single_item_to_slot(
        self, value: Any, mappings: List[Dict[Text, Any]]
    ):
        slot = self.create_slot(mappings=mappings, influence_conversation=False)
        tracker = DialogueStateTracker.from_events("sender", evts=[], slots=[slot])

        slot_event = SlotSet(slot.name, value)
        tracker.update(slot_event)

        assert tracker.slots[slot.name].value == ["cat"]


class TestCategoricalSlot(SlotTestCollection):
    def create_slot(
        self, mappings: List[Dict[Text, Any]], influence_conversation: bool
    ) -> Slot:
        return CategoricalSlot(
            "test",
            mappings=mappings,
            values=[1, "two", "小于", {"three": 3}, "nOnE", "None", "null"],
            influence_conversation=influence_conversation,
        )

    # None is a special value reserved for unset slots.
    @pytest.fixture(params=[{"a": "b"}, 2, True, "asd", "🌴", None])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0, 0, 0, 0, 0, 0]),  # slot is unset
            (1, [1, 0, 0, 0, 0, 0, 0]),
            ("two", [0, 1, 0, 0, 0, 0, 0]),
            ("小于", [0, 0, 1, 0, 0, 0, 0]),
            ({"three": 3}, [0, 0, 0, 1, 0, 0, 0]),
            ("nOnE", [0, 0, 0, 0, 1, 0, 0]),
            ("None", [0, 0, 0, 0, 1, 0, 0]),  # same as for 'nOnE' (case insensivity)
            ("null", [0, 0, 0, 0, 0, 0, 1]),
            (
                rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE,
                [0, 0, 0, 0, 0, 0, 0],
            ),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestCategoricalSlotDefaultValue(SlotTestCollection):
    def create_slot(
        self, mappings: List[Dict[Text, Any]], influence_conversation: bool
    ) -> Slot:
        slot = CategoricalSlot(
            "test",
            mappings=mappings,
            values=[1, "two", "小于", {"three": 3}, "nOnE", "None", "null"],
            influence_conversation=influence_conversation,
        )
        slot.add_default_value()
        return slot

    # None is a special value reserved for unset slots.
    @pytest.fixture(params=[{"a": "b"}, 2, True, "asd", "🌴", None])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0, 0, 0, 0, 0, 0, 0]),  # slot is unset
            (1, [1, 0, 0, 0, 0, 0, 0, 0]),
            ("two", [0, 1, 0, 0, 0, 0, 0, 0]),
            ("小于", [0, 0, 1, 0, 0, 0, 0, 0]),
            ({"three": 3}, [0, 0, 0, 1, 0, 0, 0, 0]),
            ("nOnE", [0, 0, 0, 0, 1, 0, 0, 0]),
            ("None", [0, 0, 0, 0, 1, 0, 0, 0]),  # same as for 'nOnE' (case insensivity)
            ("null", [0, 0, 0, 0, 0, 0, 1, 0]),
            (
                rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE,
                [0, 0, 0, 0, 0, 0, 0, 1],
            ),
            ("unseen value", [0, 0, 0, 0, 0, 0, 0, 1]),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestAnySlot(SlotTestCollection):
    def create_slot(
        self, mappings: List[Dict[Text, Any]], influence_conversation: bool
    ) -> Slot:
        return AnySlot("test", mappings=mappings, influence_conversation=False)

    @pytest.fixture(params=["there is nothing invalid, but we need to pass something"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, []),
            ([], []),
            ({"nested": {"dict": [1, 2, 3]}}, []),
            (["asd", 1, {}], []),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param

    def test_exception_if_featurized(self, mappings: List[Dict[Text, Any]]):
        with pytest.raises(InvalidSlotConfigError):
            AnySlot("⛔️", mappings=mappings, influence_conversation=True)


def test_raises_on_invalid_slot_type():
    with pytest.raises(InvalidSlotTypeException):
        Slot.resolve_by_type("foobar")


def test_categorical_slot_ignores_none_value():
    """Checks that None can't be added as a possible value for categorical slots."""
    with pytest.warns(UserWarning) as records:
        slot = CategoricalSlot(
            name="branch", mappings=[{}], values=["Berlin", None, "San Francisco"]
        )

    assert "none" not in slot.values

    message_text = "Rasa will ignore `null` as a possible value for the 'branch' slot."
    assert any(message_text in record.message.args[0] for record in records)
