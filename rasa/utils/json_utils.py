import json
from decimal import Decimal
from typing import Any, Dict, List, Text


class DecimalEncoder(json.JSONEncoder):
    """`json.JSONEncoder` that dumps `Decimal`s as `float`s."""

    def default(self, obj: Any) -> Any:
        """Get serializable object for `o`.

        Args:
            obj: Object to serialize.

        Returns:
            `obj` converted to `float` if `o` is a `Decimals`, else the base class
            `default()` method.
        """
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


class SetEncoder(json.JSONEncoder):
    """`json.JSONEncoder` that dumps `set`s as `list`s."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def replace_floats_with_decimals(obj: Any, round_digits: int = 9) -> Any:
    """Convert all instances in `obj` of `float` to `Decimal`.

    Args:
        obj: Input object.
        round_digits: Rounding precision of `Decimal` values.

    Returns:
        Input `obj` with all `float` types replaced by `Decimal`s rounded to
        `round_digits` decimal places.
    """

    def _float_to_rounded_decimal(s: Text) -> Decimal:
        return Decimal(s).quantize(Decimal(10) ** -round_digits)

    return json.loads(json.dumps(obj), parse_float=_float_to_rounded_decimal)


def replace_decimals_with_floats(obj: Any) -> Any:
    """Convert all instances in `obj` of `Decimal` to `float`.

    Args:
        obj: A `List` or `Dict` object.

    Returns:
        Input `obj` with all `Decimal` types replaced by `float`s.
    """
    return json.loads(json.dumps(obj, cls=DecimalEncoder))


def extract_values(data: Dict, keys: List[Text]) -> Dict:
    """Extracts values for given keys from a dictionary."""
    return {key: data.get(key) for key in keys if data.get(key)}
