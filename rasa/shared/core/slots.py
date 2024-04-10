import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Text, Type

import rasa.shared.core.constants
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.constants import (
    DOCS_URL_SLOTS,
    DOCS_URL_NLU_BASED_SLOTS,
    DOCS_URL_CATEGORICAL_SLOTS,
)

logger = logging.getLogger(__name__)


class InvalidSlotTypeException(RasaException):
    """Raised if a slot type is invalid."""


class InvalidSlotConfigError(RasaException, ValueError):
    """Raised if a slot's config is invalid."""


class Slot(ABC):
    """Key-value store for storing information during a conversation."""

    @property
    @abstractmethod
    def type_name(self) -> Text:
        """Name of the type of slot."""
        ...

    def __init__(
        self,
        name: Text,
        mappings: List[Dict[Text, Any]],
        initial_value: Any = None,
        value_reset_delay: Optional[int] = None,
        influence_conversation: bool = True,
        is_builtin: bool = False,
        shared_for_coexistence: bool = False,
    ) -> None:
        """Create a Slot.

        Args:
            name: The name of the slot.
            initial_value: The initial value of the slot.
            mappings: List containing slot mappings.
            value_reset_delay: After how many turns the slot should be reset to the
                initial_value. This is behavior is currently not implemented.
            influence_conversation: If `True` the slot will be featurized and hence
                influence the predictions of the dialogue polices.
            is_builtin: `True` if the slot is a built-in slot, `False` otherwise.
                Built-in slots also encompass user writable slots (via custom actions),
                such as `return_value`.
            shared_for_coexistence: If `True` the slot is not forgotten after either
                dm1 or CALM finishes.
        """
        self.name = name
        self.mappings = mappings
        self._value = initial_value
        self.initial_value = initial_value
        self._value_reset_delay = value_reset_delay
        self.influence_conversation = influence_conversation
        self._has_been_set = False
        self.is_builtin = is_builtin
        self.shared_for_coexistence = shared_for_coexistence

    def feature_dimensionality(self) -> int:
        """How many features this single slot creates.

        Returns:
            The number of features. `0` if the slot is unfeaturized. The dimensionality
            of the array returned by `as_feature` needs to correspond to this value.
        """
        if not self.influence_conversation:
            return 0

        return self._feature_dimensionality()

    def _feature_dimensionality(self) -> int:
        """See the docstring for `feature_dimensionality`."""
        return 1

    def has_features(self) -> bool:
        """Indicate if the slot creates any features."""
        return self.feature_dimensionality() != 0

    def value_reset_delay(self) -> Optional[int]:
        """After how many turns the slot should be reset to the initial_value.

        If the delay is set to `None`, the slot will keep its value forever.
        """
        # TODO: FUTURE this needs to be implemented - slots are not reset yet
        return self._value_reset_delay

    def as_feature(self) -> List[float]:
        if not self.influence_conversation:
            return []

        return self._as_feature()

    @abstractmethod
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
        """Resets the slot's value to the initial value."""
        self.value = self.initial_value
        self._has_been_set = False

    @property
    def value(self) -> Any:
        """Gets the slot's value."""
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        """Sets the slot's value."""
        self._value = value
        self._has_been_set = True

    def has_same_coerced_value(self, other_value: Any) -> bool:
        """Checks if the coerced value of is the same as the slot value.

        Can be used to check if an update with the value would be a no-op.
        """
        return self.coerce_value(other_value) == self.value

    def coerce_value(self, value: Any) -> Any:
        """Coerces the value to the slot's type."""
        return value

    def is_valid_value(self, value: Any) -> bool:
        """Checks if the slot contains the value."""
        return True

    @property
    def has_been_set(self) -> bool:
        """Indicates if the slot's value has been set."""
        return self._has_been_set

    def __str__(self) -> Text:
        return f"{self.__class__.__name__}({self.name}: {self.value})"

    def __repr__(self) -> Text:
        return f"<{self.__class__.__name__}({self.name}: {self.value})>"

    @staticmethod
    def resolve_by_type(type_name: Text) -> Type["Slot"]:
        """Returns a slots class by its type name."""
        for cls in rasa.shared.utils.common.all_subclasses(Slot):
            if cls.type_name == type_name:
                return cls
        try:
            return rasa.shared.utils.common.class_from_module_path(type_name)
        except (ImportError, AttributeError):
            raise InvalidSlotTypeException(
                f"Failed to find slot type, '{type_name}' is neither a known type nor "
                f"user-defined. If you are creating your own slot type, make "
                f"sure its module path is correct. "
                f"You can find all build in types at {DOCS_URL_SLOTS}"
            )

    def persistence_info(self) -> Dict[str, Any]:
        """Returns relevant information to persist this slot."""
        return {
            "type": rasa.shared.utils.common.module_path_from_instance(self),
            "initial_value": self.initial_value,
            "influence_conversation": self.influence_conversation,
            "mappings": self.mappings,
        }

    def fingerprint(self) -> Text:
        """Returns a unique hash for the slot which is stable across python runs.

        Returns:
            fingerprint of the slot
        """
        data = {"slot_name": self.name, "slot_value": self.value}
        data.update(self.persistence_info())
        return rasa.shared.utils.io.get_dictionary_fingerprint(data)


class FloatSlot(Slot):
    """A slot storing a float value."""

    type_name = "float"

    def __init__(
        self,
        name: Text,
        mappings: List[Dict[Text, Any]],
        initial_value: Optional[float] = None,
        value_reset_delay: Optional[int] = None,
        max_value: float = 1.0,
        min_value: float = 0.0,
        influence_conversation: bool = True,
        is_builtin: bool = False,
        shared_for_coexistence: bool = False,
    ) -> None:
        """Creates a FloatSlot.

        Raises:
            InvalidSlotConfigError, if the min-max range is invalid.
            UserWarning, if initial_value is outside the min-max range.
        """
        super().__init__(
            name,
            mappings,
            initial_value,
            value_reset_delay,
            influence_conversation,
            is_builtin,
            shared_for_coexistence,
        )
        self.max_value = max_value
        self.min_value = min_value

        if min_value >= max_value:
            raise InvalidSlotConfigError(
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
            return [1.0, (capped_value - self.min_value) / covered_range]
        except (TypeError, ValueError):
            return [0.0, 0.0]

    def persistence_info(self) -> Dict[Text, Any]:
        """Returns relevant information to persist this slot."""
        d = super().persistence_info()
        d["max_value"] = self.max_value
        d["min_value"] = self.min_value
        return d

    def coerce_value(self, value: Any) -> Any:
        """Coerces the value to the slot's type."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    def is_valid_value(self, value: Any) -> bool:
        """Checks if the slot contains the value."""
        # check that coerced type is float
        return value is None or isinstance(self.coerce_value(value), float)

    def _feature_dimensionality(self) -> int:
        return len(self.as_feature())


class BooleanSlot(Slot):
    """A slot storing a truth value."""

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

    def coerce_value(self, value: Any) -> Any:
        """Coerces the value to the slot's type."""
        try:
            return bool_from_any(value)
        except (ValueError, TypeError):
            return value

    def is_valid_value(self, value: Any) -> bool:
        """Checks if the slot contains the value."""
        return value is None or isinstance(self.coerce_value(value), bool)


def bool_from_any(x: Any) -> bool:
    """Converts bool/float/int/str to bool or raises error."""
    if isinstance(x, bool):
        return x
    elif isinstance(x, (float, int)):
        return x == 1.0
    elif isinstance(x, str):
        if x.isnumeric():
            return float(x) == 1.0
        elif x.strip().lower() == "true":
            return True
        elif x.strip().lower() == "no":
            return False
        elif x.strip().lower() == "false":
            return False
        elif x.strip().lower() == "yes":
            return True
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

    def coerce_value(self, value: Any) -> Any:
        """Coerces the value to the slot's type."""
        if value and not isinstance(value, list):
            # Make sure we always store list items
            value = [value]
        return value

    # FIXME: https://github.com/python/mypy/issues/8085
    @Slot.value.setter  # type: ignore[attr-defined,misc]
    def value(self, value: Any) -> None:
        """Sets the slot's value."""
        # Call property setter of superclass
        # FIXME: https://github.com/python/mypy/issues/8085
        super(ListSlot, self.__class__).value.fset(self, self.coerce_value(value))  # type: ignore[attr-defined]


class CategoricalSlot(Slot):
    """Slot type which can be used to branch conversations based on its value."""

    type_name = "categorical"

    def __init__(
        self,
        name: Text,
        mappings: List[Dict[Text, Any]],
        values: Optional[List[Any]] = None,
        initial_value: Any = None,
        value_reset_delay: Optional[int] = None,
        influence_conversation: bool = True,
        is_builtin: bool = False,
        shared_for_coexistence: bool = False,
    ) -> None:
        """Creates a `Categorical  Slot` (see parent class for detailed docstring)."""
        super().__init__(
            name,
            mappings,
            initial_value,
            value_reset_delay,
            influence_conversation,
            is_builtin,
            shared_for_coexistence,
        )
        if values and None in values:
            rasa.shared.utils.io.raise_warning(
                f"Categorical slot '{self.name}' has `null` listed as a possible value"
                f" in the domain file, which translates to `None` in Python. This value"
                f" is reserved for when the slot is not set, and should not be listed"
                f" as a value in the slot's definition."
                f" Rasa will ignore `null` as a possible value for the '{self.name}'"
                f" slot. Consider changing this value in your domain file to, for"
                f" example, `unset`, or provide the value explicitly as a string by"
                f' using quotation marks: "null".',
                category=UserWarning,
            )
        self.values = [str(v) for v in values if v is not None] if values else []
        self._warn_for_duplicate_coerced_values(values)

    def _warn_for_duplicate_coerced_values(
        self, possible_values: Optional[List[Any]]
    ) -> None:
        """If multiple values are coerced to the same value, warn the user.

        A warning indicates that multiple values are coerced to the same value
        which can create unexpected behavior. For example, if the slot is
        configured with values `["a", "A"]` both of these values are coerced
        to `"a"`. This can lead to unexpected behavior if the user sets the
        slot to `"A"` and the slot is later checked for the value `"a"`.

        Args:
            possible_values: The values for this slot configured in the domain.

        Returns:
            None
        """
        if not possible_values:
            return

        coerced_values: Dict[str, Any] = {}
        for value in possible_values:
            if value is None:
                continue

            if isinstance(value, Text):
                coerced_value = value.casefold()
            else:
                # Make sure we always store items as strings - though these
                # could be dicts, so we do not want to lower them
                coerced_value = str(value)

            if coerced_value in coerced_values:
                rasa.shared.utils.io.raise_warning(
                    f"Multiple values are coerced to the same value for the "
                    f"categorical slot '{self.name}'. "
                    f"Coerced value '{coerced_value}' is the same for "
                    f"values '{value}' and '{coerced_values[coerced_value]}'. "
                    f"Make sure to provide unique values for the slot.",
                    category=UserWarning,
                    docs=DOCS_URL_CATEGORICAL_SLOTS,
                )
            coerced_values[coerced_value] = value

    def add_default_value(self) -> None:
        """Adds the special default value to the list of possible values."""
        values = set(self.values)
        if rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE not in values:
            self.values.append(
                rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
            )

    def persistence_info(self) -> Dict[Text, Any]:
        """Returns serialized slot."""
        d = super().persistence_info()
        d["values"] = [
            value
            for value in self.values
            # Don't add default slot when persisting it.
            # We'll re-add it on the fly when creating the domain.
            if value != rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
        ]
        return d

    def _as_feature(self) -> List[float]:
        r = [0.0] * self.feature_dimensionality()

        # Return the zero-filled array if the slot is unset (i.e. set to None).
        # Conceptually, this is similar to the case when the featurisation process
        # fails, hence the returned features here are the same as for that case.
        if self.value is None:
            return r

        try:
            for i, v in enumerate(self.values):
                if v.lower() == str(self.value).lower():
                    r[i] = 1.0
                    break
            else:
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

    def coerce_value(self, value: Any) -> Any:
        """Coerces the value to the slot's configured values if possible.

        If the value can not be mapped to a configured value, the value
        is returned as is.

        String values are compared case insensitive. For example, if the
        slot is configured with values `["a", "b"]` and the value is `"A"`,
        the value is coerced to `"a"`. One exception is the case where
        the exact value is configured in the domain. In this case, the value's
        casing is not changed. For example, if the slot is configured with
        values `["A", "B"]` and the value is `"A"`, the value is coerced to
        `"A"`.

        Args:
            value: The value to coerce.

        Returns:
            The coerced value or the value as is.
        """
        if value and isinstance(value, Text):
            # Make sure we always store items with the casing
            # as they are defined in the domain
            # first, try to find an exact match
            if value in self.values:
                return value

            # fallback: try to find a case insensitive match
            for configured_value in self.values:
                if value.casefold() == configured_value.casefold():
                    return configured_value
        return value

    def is_valid_value(self, value: Any) -> bool:
        """Checks if the slots configured values contains the value.

        If the value is `None`, it is always considered valid.

        Args:
            value: The value to check.

        Returns:
            `True` if the value is valid, `False` otherwise.
        """
        if value is None:
            return True
        if isinstance(value, Text):
            # Make sure we always store items with the casing
            # as they are defined in the domain
            for configured_value in self.values:
                if value.casefold() == configured_value.casefold():
                    return True
            return False

        return str(value) in self.values

    # FIXME: https://github.com/python/mypy/issues/8085
    @Slot.value.setter  # type: ignore[attr-defined,misc]
    def value(self, value: Any) -> None:
        """Sets the slot's value after coercing it to a configured value.

        If the value can not be mapped to a configured value, the value
        is kept as is.
        """
        # Call property setter of superclass
        # FIXME: https://github.com/python/mypy/issues/8085
        super(CategoricalSlot, self.__class__).value.fset(  # type: ignore[attr-defined]
            self, self.coerce_value(value)
        )


class AnySlot(Slot):
    """Slot which can be used to store any value.

    Users need to create a subclass of `Slot` in case
    the information is supposed to get featurized.
    """

    type_name = "any"

    def __init__(
        self,
        name: Text,
        mappings: List[Dict[Text, Any]],
        initial_value: Any = None,
        value_reset_delay: Optional[int] = None,
        influence_conversation: bool = False,
        is_builtin: bool = False,
        shared_for_coexistence: bool = False,
    ) -> None:
        """Creates an `Any  Slot` (see parent class for detailed docstring).

        Raises:
            InvalidSlotConfigError, if slot is featurized.
        """
        if influence_conversation:
            raise InvalidSlotConfigError(
                f"An {AnySlot.__name__} cannot be featurized. "
                f"Please use a different slot type for slot '{name}' instead. If you "
                f"need to featurize a data type which is not supported out of the box, "
                f"implement a custom slot type by subclassing '{Slot.__name__}'. See "
                f"the documentation for more information: {DOCS_URL_NLU_BASED_SLOTS}"
            )

        super().__init__(
            name,
            mappings,
            initial_value,
            value_reset_delay,
            influence_conversation,
            is_builtin,
            shared_for_coexistence,
        )

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, AnySlot):
            return NotImplemented

        return (
            self.name == other.name
            and self.initial_value == other.initial_value
            and self._value_reset_delay == other._value_reset_delay
            and self.value == other.value
        )

    def _as_feature(self) -> List[float]:
        raise InvalidSlotConfigError(
            f"An {AnySlot.__name__} cannot be featurized. "
            f"Please use a different slot type for slot '{self.name}' instead. If you "
            f"need to featurize a data type which is not supported out of the box, "
            f"implement a custom slot type by subclassing '{Slot.__name__}'. "
            f"See the documentation for more information: {DOCS_URL_NLU_BASED_SLOTS}"
        )
