# coding=utf-8
import pytest

from rasa.core.slots import *


class SlotTestCollection(object):
    """Tests every slot needs to fulfill.

    Each slot can declare further tests on its own."""

    def create_slot(self):
        raise NotImplementedError

    def value_feature_pair(self, request):
        """Values where featurization is defined and should be tested."""
        raise NotImplementedError

    def invalid_value(self, request):
        """Values, that should be handled gracefully but where the
        featurization is not defined."""

        raise NotImplementedError

    def test_featurization(self, value_feature_pair):
        slot = self.create_slot()
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

    def test_empty_slot_featurization(self):
        slot = self.create_slot()
        assert (
            slot.value == slot.initial_value
        ), "An empty slot should be set to the initial value"
        assert len(slot.as_feature()) == slot.feature_dimensionality()

    def test_has_a_type_name(self):
        slot = self.create_slot()
        assert slot.type_name is not None
        assert type(slot) == Slot.resolve_by_type(slot.type_name)

    def test_handles_invalid_values(self, invalid_value):
        slot = self.create_slot()
        slot.value = invalid_value
        assert slot.as_feature() is not None
        assert len(slot.as_feature()) == slot.feature_dimensionality()


class TestTextSlot(SlotTestCollection):
    def create_slot(self):
        return TextSlot("test")

    @pytest.fixture(params=[1, {"a": "b"}, 2.0, [], True])
    def invalid_value(self, request):
        return request.param

    @pytest.fixture(
        params=[
            (None, [0]),
            ("", [1]),
            ("some test string", [1]),
            ("some test string 🌴", [1]),
        ]
    )
    def value_feature_pair(self, request):
        return request.param


class TestBooleanSlot(SlotTestCollection):
    def create_slot(self):
        return BooleanSlot("test")

    @pytest.fixture(params=[{"a": "b"}, [], "asd", "🌴"])
    def invalid_value(self, request):
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0]),
            (True, [1, 1]),
            ("9", [1, 1]),
            (12, [1, 1]),
            (False, [1, 0]),
            ("0", [1, 0]),
            (0, [1, 0]),
        ]
    )
    def value_feature_pair(self, request):
        return request.param


class TestFloatSlot(SlotTestCollection):
    def create_slot(self):
        return FloatSlot("test")

    @pytest.fixture(params=[{"a": "b"}, [], "asd", "🌴"])
    def invalid_value(self, request):
        return request.param

    @pytest.fixture(
        params=[
            (None, [0]),
            (True, [1]),
            (2.0, [1]),
            (1.0, [1]),
            (0.5, [0.5]),
            (0, [0]),
            (-0.5, [0.0]),
        ]
    )
    def value_feature_pair(self, request):
        return request.param


class TestListSlot(SlotTestCollection):
    def create_slot(self):
        return ListSlot("test")

    @pytest.fixture(params=[{"a": "b"}, 1, True, "asd", "🌴"])
    def invalid_value(self, request):
        return request.param

    @pytest.fixture(params=[(None, [0]), ([], [0]), ([1], [1]), (["asd", 1, {}], [1])])
    def value_feature_pair(self, request):
        return request.param


class TestUnfeaturizedSlot(SlotTestCollection):
    def create_slot(self):
        return UnfeaturizedSlot("test")

    @pytest.fixture(params=["there is nothing invalid, but we need to pass something"])
    def invalid_value(self, request):
        return request.param

    @pytest.fixture(params=[(None, []), ([23], []), (1, []), ("asd", [])])
    def value_feature_pair(self, request):
        return request.param


class TestCategoricalSlot(SlotTestCollection):
    def create_slot(self):
        return CategoricalSlot("test", values=[1, "two", "小于", {"three": 3}, None])

    @pytest.fixture(params=[{"a": "b"}, 2, True, "asd", "🌴"])
    def invalid_value(self, request):
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0, 0, 0, 1]),
            (1, [1, 0, 0, 0, 0]),
            ("two", [0, 1, 0, 0, 0]),
            ("小于", [0, 0, 1, 0, 0]),
            ({"three": 3}, [0, 0, 0, 1, 0]),
        ]
    )
    def value_feature_pair(self, request):
        return request.param
