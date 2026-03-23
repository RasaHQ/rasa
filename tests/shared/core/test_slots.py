import re
from typing import Any, Dict, List, Optional, Text, Tuple

import pytest
from _pytest.fixtures import SubRequest

import rasa.shared.core.constants
from rasa.shared.constants import REFILL_UTTER, REJECTIONS
from rasa.shared.core.events import SlotSet
from rasa.shared.core.slots import (
    AnySlot,
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    InvalidSlotConfigError,
    InvalidSlotTypeException,
    InvalidSlotValueError,
    ListSlot,
    Slot,
    StrictCategoricalSlot,
    TextSlot,
    bool_from_any,
)
from rasa.shared.core.trackers import DialogueStateTracker


class SlotTestCollection:
    """Tests every slot needs to fulfill.

    Each slot can declare further tests on its own.
    """

    def create_slot(
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool,
        validation: Optional[Dict[Text, Any]] = None,
    ) -> Slot:
        raise NotImplementedError

    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        """Values where featurization is defined and should be tested."""
        raise NotImplementedError

    def invalid_value(self, request: SubRequest) -> Any:
        """Values, that should be handled gracefully but where the
        featurization is not defined.
        """
        raise NotImplementedError

    @pytest.fixture()
    def mappings(self) -> List[Dict[Text, Any]]:
        return [{}]

    @pytest.fixture()
    def validation(self) -> Dict[Text, Any]:
        return {
            REFILL_UTTER: "utter_test_slot",
            REJECTIONS: [
                {"if": "test_slot == 'invalid'", "utter": "utter_invalid_test_slot"}
            ],
        }

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

    def test_none_is_valid_value(self, mappings: List[Dict[Text, Any]]):
        slot = self.create_slot(mappings=mappings, influence_conversation=False)
        assert slot.is_valid_value(None)

    def test_none_stays_none_when_coercing(self, mappings: List[Dict[Text, Any]]):
        slot = self.create_slot(mappings=mappings, influence_conversation=False)
        assert slot.coerce_value(None) is None

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

    def test_slot_is_not_builtin_by_default(self, mappings: List[Dict[Text, Any]]):
        slot = self.create_slot(mappings, influence_conversation=False)
        assert not slot.is_builtin

    def test_slot_has_validation(self, validation: Dict[Text, Any]):
        slot = self.create_slot(
            mappings=[], influence_conversation=True, validation=validation
        )
        assert slot.requires_validation


class TestTextSlot(SlotTestCollection):
    def create_slot(
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool,
        validation: Optional[Dict[Text, Any]] = None,
    ) -> Slot:
        return TextSlot(
            "test",
            mappings=mappings,
            influence_conversation=influence_conversation,
            validation=validation,
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
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool,
        validation: Optional[Dict[Text, Any]] = None,
    ) -> Slot:
        return BooleanSlot(
            "test",
            mappings=mappings,
            influence_conversation=influence_conversation,
            validation=validation,
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
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool = False,
        validation: Optional[Dict[Text, Any]] = None,
        initial_value: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Slot:
        return FloatSlot(
            "test",
            mappings=mappings,
            influence_conversation=influence_conversation,
            validation=validation,
            initial_value=initial_value,
            min_value=min_value,
            max_value=max_value,
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

    @pytest.mark.parametrize(
        "min_value, max_value, error_msg",
        [
            (
                5.0,
                1.0,
                "Float slot ('test') created with an invalid range "
                "using min (5.0) and max (1.0) values. ",
            ),
            (
                1.0,
                1.0,
                "Float slot ('test') created with an invalid range "
                "using min (1.0) and max (1.0) values. ",
            ),
        ],
    )
    def test_validate_min_greater_or_equal_to_max_raises(
        self, min_value: float, max_value: float, error_msg: str
    ) -> None:
        error_msg += "Make sure min is smaller than max."
        with pytest.raises(InvalidSlotConfigError, match=re.escape(error_msg)):
            self.create_slot(mappings=[], min_value=min_value, max_value=max_value)

    @pytest.mark.parametrize(
        "initial_value, error_msg",
        [
            (6.0, "Float slot ('test') created with an initial value 6.0. "),
            (1.0, "Float slot ('test') created with an initial value 1.0. "),
        ],
    )
    def test_validate_min_max_range_of_initial_value_raises(
        self, initial_value: float, error_msg: str
    ) -> None:
        error_msg += (
            "This value is outside of the configured min (1.5) and max (5.0) values."
        )
        with pytest.raises(InvalidSlotConfigError, match=re.escape(error_msg)):
            self.create_slot(
                mappings=[], initial_value=initial_value, min_value=1.5, max_value=5.0
            )

    def test_is_valid_value(self):
        slot = self.create_slot(mappings=[], min_value=1.5, max_value=5.0)
        assert slot.is_valid_value(1.5)
        assert slot.is_valid_value(3.0)
        assert slot.is_valid_value(5.0)
        assert not slot.is_valid_value(1.4)
        assert not slot.is_valid_value(5.1)

    def test_as_feature(self):
        slot = self.create_slot(
            mappings=[], min_value=1.5, max_value=5.0, influence_conversation=True
        )
        slot.value = 3.0
        assert slot.as_feature() == [1, (3.0 - 1.5) / (5.0 - 1.5)]
        slot.value = 1.5
        assert slot.as_feature() == [1, 0.0]
        slot.value = 5.0
        assert slot.as_feature() == [1, 1.0]
        slot.value = 1.4
        assert slot.as_feature() == [1, 0.0]


class TestListSlot(SlotTestCollection):
    def create_slot(
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool,
        validation: Optional[Dict[Text, Any]] = None,
    ) -> Slot:
        return ListSlot(
            "test",
            mappings=mappings,
            influence_conversation=influence_conversation,
            validation=validation,
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
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool,
        validation: Optional[Dict[Text, Any]] = None,
    ) -> Slot:
        return CategoricalSlot(
            "test",
            mappings=mappings,
            values=[1, "two", "小于", {"three": 3}, "nOnE", "None", "null"],
            influence_conversation=influence_conversation,
            validation=validation,
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

    @pytest.mark.parametrize(
        "value", [1, "two", "小于", {"three": 3}, "nOnE", "None", "null", None]
    )
    def test_is_valid_value(self, value):
        slot = self.create_slot(mappings=[], influence_conversation=False)
        assert slot.is_valid_value(value)

    def test_is_invalid_value(self):
        slot = self.create_slot(mappings=[], influence_conversation=False)
        assert not slot.is_valid_value("unseen value")

    def test_setting_value_coerces_case_lowercase(self):
        slot = self.create_slot(mappings=[], influence_conversation=False)
        slot.value = "TWO"
        assert slot.value == "two"

    def test_setting_value_coerces_case_uppercase(self):
        slot = self.create_slot(mappings=[], influence_conversation=False)
        slot.value = "none"
        assert slot.value == "nOnE"

    def test_setting_value_coerces_case_only_if_exact_match_is_not_found(self):
        slot = self.create_slot(mappings=[], influence_conversation=False)
        slot.value = "nOnE"
        assert slot.value == "nOnE"

        slot.value = "None"
        assert slot.value == "None"

    def test_stores_value_as_if_not_found(self):
        slot = self.create_slot(mappings=[], influence_conversation=False)
        slot.value = "unseen value"
        assert slot.value == "unseen value"

    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, None),
            (1, 1),
            ("two", "two"),
            ("TWO", "two"),
            ("小于", "小于"),
            ({"three": 3}, {"three": 3}),
            ("nOnE", "nOnE"),
            ("None", "None"),
            ("null", "null"),
            ("unseen value", "unseen value"),
        ],
    )
    def test_coerces_values(self, value: Any, expected: Any):
        slot = self.create_slot(mappings=[], influence_conversation=False)
        assert slot.coerce_value(value) == expected

    def test_raises_warning_on_coerced_duplicate(self):
        with pytest.warns(UserWarning) as records:
            CategoricalSlot(
                "test",
                mappings=[],
                values=[1, "two", "TWO"],
            )

        assert len(records) == 1
        assert (
            "Multiple values are coerced to the same value"
            in records[0].message.args[0]
        )


class TestCategoricalSlotDefaultValue(SlotTestCollection):
    def create_slot(
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool,
        validation: Optional[Dict[Text, Any]] = None,
    ) -> Slot:
        slot = CategoricalSlot(
            "test",
            mappings=mappings,
            values=[1, "two", "小于", {"three": 3}, "nOnE", "None", "null"],
            influence_conversation=influence_conversation,
            validation=validation,
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
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool,
        validation: Optional[Dict[Text, Any]] = None,
    ) -> Slot:
        return AnySlot(
            "test",
            mappings=mappings,
            influence_conversation=False,
            validation=validation,
        )

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


class TestStrictCategoricalSlot(SlotTestCollection):
    def create_slot(
        self,
        mappings: List[Dict[Text, Any]],
        influence_conversation: bool,
        validation: Optional[Dict[Text, Any]] = None,
    ) -> Slot:
        # Use a fixed list of allowed values as strings.
        # The order determines the one-hot encoding:
        # "1"    → [1, 0, 0, 0]
        # "two"  → [0, 1, 0, 0]
        # "three"→ [0, 0, 1, 0]
        # "nOnE" → [0, 0, 0, 1]
        # (Coercion will normalize case as defined.)
        return StrictCategoricalSlot(
            "test",
            mappings=mappings,
            values=["1", "two", "three", "nOnE"],
            influence_conversation=influence_conversation,
            validation=validation,
        )

    @pytest.fixture(
        params=[
            (None, [0, 0, 0, 0]),  # unset slot: no value chosen
            ("1", [1, 0, 0, 0]),
            ("two", [0, 1, 0, 0]),
            ("TWO", [0, 1, 0, 0]),  # valid: normalized to "two"
            ("three", [0, 0, 1, 0]),
            ("None", [0, 0, 0, 1]),  # valid: matches "nOnE"
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param

    @pytest.fixture(params=["unseen", "999", "invalid"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    # Override the base test for invalid values, so it expects an exception.
    def test_handles_invalid_values(
        self, invalid_value: Any, mappings: List[Dict[Text, Any]]
    ):
        slot = self.create_slot(mappings=mappings, influence_conversation=True)
        with pytest.raises(InvalidSlotValueError):
            slot.value = invalid_value

    # Override fingerprint uniqueness test so that it uses an allowed value change.
    @pytest.mark.parametrize("influence_conversation", [True, False])
    def test_slot_fingerprint_uniqueness(
        self, influence_conversation: bool, mappings: List[Dict[Text, Any]]
    ):
        slot = self.create_slot(mappings, influence_conversation)
        f1 = slot.fingerprint()
        slot.value = "1"
        f2 = slot.fingerprint()
        assert f1 != f2

    def test_set_invalid_value_raises_error(self, mappings: List[Dict[Text, Any]]):
        slot = self.create_slot(mappings=mappings, influence_conversation=False)
        with pytest.raises(InvalidSlotValueError):
            slot.value = "unseen"

    def test_strict_coercion_normalizes_valid_value(
        self, mappings: List[Dict[Text, Any]]
    ):
        slot = self.create_slot(mappings=mappings, influence_conversation=False)
        # Set a value that is valid but in the wrong case. It should be normalized.
        slot.value = "TWO"
        assert slot.value == "two"

    def test_can_reset_to_none_once_set(self, mappings: List[Dict[Text, Any]]):
        # StrictCategoricalSlot allows resetting to None even after a value is set.
        slot = self.create_slot(mappings=mappings, influence_conversation=False)
        slot.value = "1"
        slot.value = None
        assert slot.value is None
