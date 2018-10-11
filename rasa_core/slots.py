from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import str

import logging

from rasa_core import utils

logger = logging.getLogger(__name__)


class Slot(object):
    type_name = None

    def __init__(self, name, initial_value=None, value_reset_delay=None):
        self.name = name
        self.value = initial_value
        self.initial_value = initial_value
        self._value_reset_delay = value_reset_delay

    def feature_dimensionality(self):
        """How many features this single slot creates.

        The dimensionality of the array returned by `as_feature` needs
        to correspond to this value."""
        return 1

    def has_features(self):
        """Indicate if the slot creates any features."""
        return self.feature_dimensionality() != 0

    def value_reset_delay(self):
        """After how many turns the slot should be reset to the initial_value.

        If the delay is set to `None`, the slot will keep its value forever."""
        # TODO: FUTURE this needs to be implemented - slots are not reset yet
        return self._value_reset_delay

    def as_feature(self):
        raise NotImplementedError("Each slot type needs to specify how its "
                                  "value can be converted to a feature. Slot "
                                  "'{}' is a generic slot that can not be used "
                                  "for predictions. Make sure you add this "
                                  "slot to your domain definition, specifying "
                                  "the type of the slot. If you implemented "
                                  "a custom slot type class, make sure to "
                                  "implement `.as_feature()`."
                                  "".format(self.name))

    def reset(self):
        self.value = self.initial_value

    def __str__(self):
        return "{}({}: {})".format(self.__class__.__name__,
                                   self.name,
                                   self.value)

    def __repr__(self):
        return "<{}({}: {})>".format(self.__class__.__name__,
                                     self.name,
                                     self.value)

    @staticmethod
    def resolve_by_type(type_name):
        """Returns a slots class by its type name."""

        for cls in utils.all_subclasses(Slot):
            if cls.type_name == type_name:
                return cls
        try:
            return utils.class_from_module_path(type_name)
        except Exception:
            raise ValueError(
                    "Failed to find slot type. Neither a known type nor. If "
                    "you are creating your own slot type, make sure its "
                    "module path is correct: {}.".format(type_name))

    def persistence_info(self):
        return {"type": utils.module_path_from_instance(self),
                "initial_value": self.initial_value}


class FloatSlot(Slot):
    type_name = "float"

    def __init__(self, name,
                 initial_value=None,
                 value_reset_delay=None,
                 max_value=1.0,
                 min_value=0.0):
        super(FloatSlot, self).__init__(name, initial_value, value_reset_delay)
        self.max_value = max_value
        self.min_value = min_value

        if min_value >= max_value:
            raise ValueError(
                    "Float slot ('{}') created with an invalid range "
                    "using min ({}) and max ({}) values. Make sure "
                    "min is smaller than max."
                    "".format(self.name, self.min_value, self.max_value))

        if (initial_value is not None and
                not (min_value <= initial_value <= max_value)):
            logger.warning("Float slot ('{}') created with an initial value {}"
                           "outside of configured min ({}) and max ({}) values."
                           "".format(self.name, self.value, self.min_value,
                                     self.max_value))

    def as_feature(self):
        try:
            capped_value = max(self.min_value,
                               min(self.max_value, float(self.value)))
            if abs(self.max_value - self.min_value) > 0:
                covered_range = abs(self.max_value - self.min_value)
            else:
                covered_range = 1
            return [(capped_value - self.min_value) / covered_range]
        except (TypeError, ValueError):
            return [0.0]

    def persistence_info(self):
        d = super(FloatSlot, self).persistence_info()
        d["max_value"] = self.max_value
        d["min_value"] = self.min_value
        return d


class BooleanSlot(Slot):
    type_name = "bool"

    def as_feature(self):
        try:
            if self.value is not None:
                return [1.0, float(float(self.value) != 0.0)]
            else:
                return [0.0, 0.0]
        except (TypeError, ValueError):
            # we couldn't convert the value to float - using default value
            return [0.0, 0.0]

    def feature_dimensionality(self):
        return len(self.as_feature())


class TextSlot(Slot):
    type_name = "text"

    def as_feature(self):
        return [1.0 if self.value is not None else 0.0]


class ListSlot(Slot):
    type_name = "list"

    def as_feature(self):
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

    def as_feature(self):
        return []

    def feature_dimensionality(self):
        return 0


class CategoricalSlot(Slot):
    type_name = "categorical"

    def __init__(self, name,
                 values=None,
                 initial_value=None,
                 value_reset_delay=None):
        super(CategoricalSlot, self).__init__(name,
                                              initial_value,
                                              value_reset_delay)
        self.values = [str(v).lower() for v in values] if values else []

    def persistence_info(self):
        d = super(CategoricalSlot, self).persistence_info()
        d["values"] = self.values
        return d

    def as_feature(self):
        r = [0.0] * self.feature_dimensionality()

        try:
            for i, v in enumerate(self.values):
                if v == str(self.value).lower():
                    r[i] = 1.0
                    break
            else:
                if self.value is not None:
                    logger.warning(
                            "Categorical slot '{}' is set to a value ('{}') "
                            "that is not specified in the domain. "
                            "Value will be ignored and the slot will "
                            "behave as if no value is set. "
                            "Make sure to add all values a categorical "
                            "slot should store to the domain."
                            "".format(self.name, self.value))
        except (TypeError, ValueError):
            logger.exception("Failed to featurize categorical slot.")
            return r
        return r

    def feature_dimensionality(self):
        return len(self.values)


class DataSlot(Slot):
    def __init__(self, name, initial_value=None, value_reset_delay=1):
        super(DataSlot, self).__init__(name, initial_value, value_reset_delay)

    def as_feature(self):
        raise NotImplementedError("Each slot type needs to specify how its "
                                  "value can be converted to a feature.")
