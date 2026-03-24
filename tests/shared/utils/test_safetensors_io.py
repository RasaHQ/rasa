import builtins
from typing import Any, Optional, Tuple

import pytest

from rasa.exceptions import MissingDependencyException
from rasa.shared.utils.safetensors_io import safetensors_numpy_load_save


def _block_safetensors_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: Optional[dict] = None,
        locals_: Optional[dict] = None,
        fromlist: Tuple[Any, ...] = (),
        level: int = 0,
    ):
        if name == "safetensors.numpy":
            raise ImportError("simulated missing safetensors")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_safetensors_numpy_load_save_default_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_safetensors_numpy(monkeypatch)
    with pytest.raises(MissingDependencyException) as exc_info:
        safetensors_numpy_load_save()
    msg = str(exc_info.value)
    assert "safetensors" in msg
    assert "rasa-pro[nlu]" in msg


def test_safetensors_numpy_load_save_custom_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_safetensors_numpy(monkeypatch)
    custom = "Custom prefix: install rasa-pro[nlu]"
    with pytest.raises(MissingDependencyException) as exc_info:
        safetensors_numpy_load_save(missing_dependency_message=custom)
    assert str(exc_info.value) == custom
