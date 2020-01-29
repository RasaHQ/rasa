import logging
from typing import Any, Dict, List, NoReturn, Optional, Text, Type

from rasa.core import utils
from rasa.utils.common import class_from_module_path, raise_warning

logger = logging.getLogger(__name__)


class Slot:
    type_name = None

    def __init__(
        self,
        name: Text,
        initial_value: Any = None,
        value_reset_delay: Optional[int] = None,
        auto_fill: bool = True,
    ) -> None:
        self.name = name
        self.value = initial_value
        self.initial_value = initial_value
        self._value_reset_delay = value_reset_delay
        self.auto_fill = auto_fill

    def feature_dimensionality(self) -> int:
        """How many features this single slot creates.

        The dimensionality of the array returned by `as_feature` needs
        to correspond to this value."""
        return 1

    def has_features(self) -> bool:
        """Indicate if the slot creates any features."""
        return self.feature_dimensionality() != 0

    def value_reset_delay(self) -> Optional[int]:
        """After how many turns the slot should be reset to the initial_value.

        If the delay is set to `None`, the slot will keep its value forever."""
        # TODO: FUTURE this needs to be implemented - slots are not reset yet
        return self._value_reset_delay

    def as_feature(self) -> NoReturn:
        raise NotImplementedError(
            "Each slot type needs to specify how its "
            "value can be converted to a feature. Slot "
            "'{}' is a generic slot that can not be used "
            "for predictions. Make sure you add this "
            "slot to your domain definition, specifying "
            "the type of the slot. If you implemented "
            "a custom slot type class, make sure to "
            "implement `.as_feature()`."
            "".format(self.name)
        )

    def reset(self) -> None:
        self.value = self.initial_value

    def __str__(self) -> Text:
        return f"{self.__class__.__name__}({self.name}: {self.value})"

    def __repr__(self) -> Text:
        return f"<{self.__class__.__name__}({self.name}: {self.value})>"

    @staticmethod
    def resolve_by_type(type_name) -> Type["Slot"]:
        """Returns a slots class by its type name."""
        for cls in utils.all_subclasses(Slot):
            if cls.type_name == type_name:
                return cls
        try:
            return class_from_module_path(type_name)
        except (ImportError, AttributeError):
            raise ValueError(
                "Failed to find slot type, '{}' is neither a known type nor "
                "user-defined. If you are creating your own slot type, make "
                "sure its module path is correct.".format(type_name)
            )

    def persistence_info(self) -> Dict[str, Any]:
        return {
            "type": utils.module_path_from_instance(self),
            "initial_value": self.initial_value,
            "auto_fill": self.auto_fill,
        }


class FloatSlot(Slot):
    type_name = "float"

    def __init__(
        self,
        name: Text,
        initial_value: Optional[float] = None,
        value_reset_delay: Optional[int] = None,
        auto_fill: bool = True,
        max_value: float = 1.0,
        min_value: float = 0.0,
    ) -> None:
        super().__init__(name, initial_value, value_reset_delay, auto_fill)
        self.max_value = max_value
        self.min_value = min_value

        if min_value >= max_value:
            raise ValueError(
                "Float slot ('{}') created with an invalid range "
                "using min ({}) and max ({}) values. Make sure "
                "min is smaller than max."
                "".format(self.name, self.min_value, self.max_value)
            )

        if initial_value is not None and not (min_value <= initial_value <= max_value):
            raise_warning(
                f"Float slot ('{self.name}') created with an initial value "
                f"{self.value}. This value is outside of the configured min "
                f"({self.min_value}) and max ({self.max_value}) values."
            )

    def as_feature(self) -> List[float]:
        try:
            capped_value = max(self.min_value, min(self.max_value, float(self.value)))
            if abs(self.max_value - self.min_value) > 0:
                covered_range = abs(self.max_value - self.min_value)
            else:
                covered_range = 1
            return [(capped_value - self.min_value) / covered_range]
        except (TypeError, ValueError):
            return [0.0]

    def persistence_info(self) -> Dict[Text, Any]:
        d = super().persistence_info()
        d["max_value"] = self.max_value
        d["min_value"] = self.min_value
        return d


class BooleanSlot(Slot):
    type_name = "bool"

    def as_feature(self) -> List[float]:
        try:
            if self.value is not None:
                return [1.0, float(float(self.value) != 0.0)]
            else:
                return [0.0, 0.0]
        except (TypeError, ValueError):
            # we couldn't convert the value to float - using default value
            return [0.0, 0.0]

    def feature_dimensionality(self) -> int:
        return len(self.as_feature())


class TextSlot(Slot):
    type_name = "text"

    def as_feature(self) -> List[float]:
        return [1.0 if self.value is not None else 0.0]


class ListSlot(Slot):
    type_name = "list"

    def as_feature(self) -> List[float]:
        try:
            if self.value is not None and len(self.value) > 0:
                return [1.0]
            else:
                return [0.0]
        except (TypeError, ValueError):
            # we couldn't convert the value to a list - using default value
            return [0.0]


class UnfeaturizedSlot(Slot):
    type_name = "unfeaturized"

    def as_feature(self) -> List[float]:
        return []

    def feature_dimensionality(self) -> int:
        return 0


class CategoricalSlot(Slot):
    type_name = "categorical"

    def __init__(
        self,
        name: Text,
        values: Optional[List[Any]] = None,
        initial_value: Any = None,
        value_reset_delay: Optional[int] = None,
        auto_fill: bool = True,
    ) -> None:
        super().__init__(name, initial_value, value_reset_delay, auto_fill)
        self.values = [str(v).lower() for v in values] if values else []

    def persistence_info(self) -> Dict[Text, Any]:
        d = super().persistence_info()
        d["values"] = self.values
        return d

    def as_feature(self) -> List[float]:
        r = [0.0] * self.feature_dimensionality()

        try:
            for i, v in enumerate(self.values):
                if v == str(self.value).lower():
                    r[i] = 1.0
                    break
            else:
                if self.value is not None:
                    raise_warning(
                        f"Categorical slot '{self.name}' is set to a value "
                        f"('{self.value}') "
                        "that is not specified in the domain. "
                        "Value will be ignored and the slot will "
                        "behave as if no value is set. "
                        "Make sure to add all values a categorical "
                        "slot should store to the domain."
                    )
        except (TypeError, ValueError):
            logger.exception("Failed to featurize categorical slot.")
            return r
        return r

    def feature_dimensionality(self) -> int:
        return len(self.values)


class DataSlot(Slot):
    def __init__(
        self,
        name: Text,
        initial_value: Any = None,
        value_reset_delay: Optional[int] = 1,
        auto_fill: bool = True,
    ):
        super().__init__(name, initial_value, value_reset_delay, auto_fill)

    def as_feature(self) -> List[float]:
        raise NotImplementedError(
            "Each slot type needs to specify how its "
            "value can be converted to a feature."
        )
