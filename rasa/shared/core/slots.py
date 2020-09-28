import logging

from typing import Any, Dict, List, Optional, Text, Type

import rasa.shared.core.constants
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_SLOTS

logger = logging.getLogger(__name__)


class Slot:
    type_name = None

    def __init__(
        self,
        name: Text,
        initial_value: Any = None,
        value_reset_delay: Optional[int] = None,
        auto_fill: bool = True,
        unfeaturized: bool = True,
    ) -> None:
        """Create a Slot.

        Args:
            name: The name of the slot.
            initial_value: The initial value of the slot.
            value_reset_delay: After how many turns the slot should be reset to the
                initial_value. This is behavior is currently not implemented.
            auto_fill: `True` if the slot should be filled automatically by entities
                with the same name.
            unfeaturized: If `True` the slot will not be featurized and hence not affect
                the predictions of dialogue polices.
        """
        self.name = name
        self.value = initial_value
        self.initial_value = initial_value
        self._value_reset_delay = value_reset_delay
        self.auto_fill = auto_fill
        self.unfeaturized = unfeaturized

    def feature_dimensionality(self) -> int:
        """How many features this single slot creates.

        Returns:
            The number of features. `0` if the slot is unfeaturized. The dimensionality
            of the array returned by `as_feature` needs to correspond to this value.
        """
        if self.unfeaturized:
            return 0

        return self._feature_dimensionality()

    def _feature_dimensionality(self) -> int:
        """See the docstring for `feature_dimensionality`."""
        return 1

    def add_default_value(self) -> None:
        """Add a default value to a slots user-defined values"""
        raise NotImplementedError(
            "Each slot type needs to specify its own" "default value to add, if any"
        )

    def has_features(self) -> bool:
        """Indicate if the slot creates any features."""
        return self.feature_dimensionality() != 0

    def value_reset_delay(self) -> Optional[int]:
        """After how many turns the slot should be reset to the initial_value.

        If the delay is set to `None`, the slot will keep its value forever."""
        # TODO: FUTURE this needs to be implemented - slots are not reset yet
        return self._value_reset_delay

    def as_feature(self) -> List[float]:
        if self.unfeaturized:
            return []

        return self._as_feature()

    def _as_feature(self) -> List[float]:
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
        for cls in rasa.shared.utils.common.all_subclasses(Slot):
            if cls.type_name == type_name:
                return cls
        try:
            return rasa.shared.utils.common.class_from_module_path(type_name)
        except (ImportError, AttributeError):
            raise ValueError(
                "Failed to find slot type, '{}' is neither a known type nor "
                "user-defined. If you are creating your own slot type, make "
                "sure its module path is correct.".format(type_name)
            )

    def persistence_info(self) -> Dict[str, Any]:
        return {
            "type": rasa.shared.utils.common.module_path_from_instance(self),
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
        unfeaturized: bool = False,
    ) -> None:
        super().__init__(
            name, initial_value, value_reset_delay, auto_fill, unfeaturized
        )
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
            rasa.shared.utils.io.raise_warning(
                f"Float slot ('{self.name}') created with an initial value "
                f"{self.value}. This value is outside of the configured min "
                f"({self.min_value}) and max ({self.max_value}) values."
            )

    def _as_feature(self) -> List[float]:
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

    def _as_feature(self) -> List[float]:
        try:
            if self.value is not None:
                return [1.0, float(bool_from_any(self.value))]
            else:
                return [0.0, 0.0]
        except (TypeError, ValueError):
            # we couldn't convert the value to float - using default value
            return [0.0, 0.0]

    def _feature_dimensionality(self) -> int:
        return len(self.as_feature())


def bool_from_any(x: Any) -> bool:
    """ Converts bool/float/int/str to bool or raises error """

    if isinstance(x, bool):
        return x
    elif isinstance(x, (float, int)):
        return x == 1.0
    elif isinstance(x, str):
        if x.isnumeric():
            return float(x) == 1.0
        elif x.strip().lower() == "true":
            return True
        elif x.strip().lower() == "false":
            return False
        else:
            raise ValueError("Cannot convert string to bool")
    else:
        raise TypeError("Cannot convert to bool")


class TextSlot(Slot):
    type_name = "text"

    def _as_feature(self) -> List[float]:
        return [1.0 if self.value is not None else 0.0]


class ListSlot(Slot):
    type_name = "list"

    def _as_feature(self) -> List[float]:
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

    def __init__(
        self,
        name: Text,
        initial_value: Any = None,
        value_reset_delay: Optional[int] = None,
        auto_fill: bool = True,
        unfeaturized: bool = True,
    ) -> None:
        if not unfeaturized:
            raise ValueError(
                f"An {UnfeaturizedSlot.__name__} cannot be featurized. "
                f"Please use a different slot type instead. See the "
                f"documentation for more information: {DOCS_URL_SLOTS}"
            )

        super().__init__(
            name, initial_value, value_reset_delay, auto_fill, unfeaturized
        )

    def _as_feature(self) -> List[float]:
        return []

    def _feature_dimensionality(self) -> int:
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
        unfeaturized: bool = False,
    ) -> None:
        super().__init__(
            name, initial_value, value_reset_delay, auto_fill, unfeaturized
        )
        self.values = [str(v).lower() for v in values] if values else []

    def add_default_value(self) -> None:
        values = set(self.values)
        if rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE not in values:
            self.values.append(
                rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
            )

    def persistence_info(self) -> Dict[Text, Any]:
        d = super().persistence_info()
        d["values"] = self.values
        return d

    def _as_feature(self) -> List[float]:
        r = [0.0] * self.feature_dimensionality()

        try:
            for i, v in enumerate(self.values):
                if v == str(self.value).lower():
                    r[i] = 1.0
                    break
            else:
                if self.value is not None:
                    if (
                        rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
                        in self.values
                    ):
                        i = self.values.index(
                            rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
                        )
                        r[i] = 1.0
                    else:
                        rasa.shared.utils.io.raise_warning(
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

    def _feature_dimensionality(self) -> int:
        return len(self.values)
