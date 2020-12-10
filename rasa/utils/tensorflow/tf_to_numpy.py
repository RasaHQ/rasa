from typing import Any, Dict, Optional
from tensorflow import Tensor


def values_to_numpy(data: Optional[Dict[Any, Any]]) -> Optional[Dict[Any, Any]]:
    """Replaces all tensorflow-tensor values with their numpy versions.

    Args:
        data: Any dictionary for which values should be converted.

    Returns:
        A dictionary identical to `data` except that tensor values are
        replaced by their corresponding numpy arrays.
    """
    if not data:
        return data

    return {key: _to_numpy_if_tensor(value) for key, value in data.items()}


def _to_numpy_if_tensor(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.numpy()
    else:
        return value
