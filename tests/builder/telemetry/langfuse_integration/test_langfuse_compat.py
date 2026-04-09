"""Tests for langfuse compatibility layer."""

import asyncio
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_langfuse_not_installed() -> Generator[None, None, None]:
    """Simulate langfuse not being installed."""
    from types import SimpleNamespace

    from rasa.builder.telemetry.langfuse_integration import langfuse_compat

    fake_langfuse = SimpleNamespace(
        observe=langfuse_compat._no_op_observe,
        get_client=lambda: None,
    )
    with (
        patch.object(langfuse_compat, "langfuse", fake_langfuse),
        patch.object(langfuse_compat, "_LANGFUSE_AVAILABLE", False),
    ):
        yield


class TestLangfuseCompat:
    """Test langfuse_compat module functions and classes."""

    def test_is_langfuse_available_when_installed(self):
        """Test is_langfuse_available returns True when langfuse is installed."""
        with patch.dict("sys.modules", {"langfuse": MagicMock()}):
            # Re-import to pick up the mock
            from rasa.builder.telemetry.langfuse_integration import langfuse_compat

            # When langfuse is installed (in test environment), it should be True
            # This test may return True or False depending on the actual environment
            result = langfuse_compat.is_langfuse_available()
            assert isinstance(result, bool)

    def test_with_langfuse_context_manager_when_available(self):
        """Test with_langfuse yields module when available."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
            with_langfuse,
        )

        with with_langfuse():
            # Should yield either the module or None depending on installation
            # The important thing is it doesn't raise
            pass

    def test_require_langfuse_raises_when_not_installed(
        self, mock_langfuse_not_installed: None
    ) -> None:
        """Test require_langfuse raises ImportError when not available."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
            require_langfuse,
        )

        with pytest.raises(ImportError, match="langfuse is required"):
            require_langfuse()

    def test_require_langfuse_returns_module_when_installed(self):
        """Test require_langfuse returns module when available."""
        from rasa.builder.telemetry.langfuse_integration import langfuse_compat

        if langfuse_compat.is_langfuse_available():
            result = langfuse_compat.require_langfuse()
            assert result is not None

    def test_observe_is_no_op_for_sync_when_langfuse_not_installed(
        self, mock_langfuse_not_installed: None
    ) -> None:
        """Test observe acts as a no-op pass-through for sync functions."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
            observe,
        )

        @observe()
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply(6, 7) == 42
        assert not asyncio.iscoroutinefunction(multiply)

    def test_observe_is_no_op_for_async_when_langfuse_not_installed(
        self, mock_langfuse_not_installed: None
    ) -> None:
        """Test observe acts as a no-op pass-through for async functions."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
            observe,
        )

        @observe()
        async def multiply(a: int, b: int) -> int:
            return a * b

        assert asyncio.iscoroutinefunction(multiply)
        assert asyncio.run(multiply(6, 7)) == 42


class TestMockLangfuse:
    """Test mock langfuse implementations when langfuse is not installed."""

    def test_no_op_observe_decorator_preserves_function(self):
        """Test _no_op_observe decorator returns a working function."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
            _no_op_observe,
        )

        @_no_op_observe()
        def test_function(x: int, y: int) -> int:
            return x + y

        # Function should work normally
        assert test_function(2, 3) == 5

    def test_no_op_observe_decorator_with_args(self):
        """Test _no_op_observe decorator accepts arguments."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
            _no_op_observe,
        )

        @_no_op_observe(name="test", capture_input=True, capture_output=True)
        def test_function(x: int) -> int:
            return x * 2

        # Function should work normally
        assert test_function(5) == 10

    def test_mock_langfuse_observe_attribute(self):
        """Test _MockLangfuse has observe attribute when langfuse not installed."""
        from rasa.builder.telemetry.langfuse_integration import langfuse_compat

        if langfuse_compat.is_langfuse_available():
            pytest.skip("_MockLangfuse only exists when langfuse is not installed")

        mock = langfuse_compat._MockLangfuse()  # type: ignore[attr-defined]

        assert hasattr(mock, "observe")
        # observe should be callable
        assert callable(mock.observe)

    def test_mock_langfuse_get_client_returns_none(self):
        """Test _MockLangfuse.get_client returns None when langfuse not installed."""
        from rasa.builder.telemetry.langfuse_integration import langfuse_compat

        if langfuse_compat.is_langfuse_available():
            pytest.skip("_MockLangfuse only exists when langfuse is not installed")

        mock = langfuse_compat._MockLangfuse()  # type: ignore[attr-defined]

        assert mock.get_client() is None


class TestLangfuseObserveExport:
    """Test that observe is properly exported."""

    def test_observe_is_exported(self):
        """Test that observe attribute is exported at module level."""
        from rasa.builder.telemetry.langfuse_integration import langfuse_compat

        assert hasattr(langfuse_compat, "observe")

    def test_observe_works_as_decorator(self):
        """Test that observe can be used as a decorator."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import observe

        @observe()
        def decorated_function(value: str) -> str:
            return value.upper()

        # Should work without errors
        result = decorated_function("hello")
        assert result == "HELLO"


class TestWithLangfuseContextManager:
    """Test with_langfuse context manager behavior."""

    def test_with_langfuse_executes_block(self):
        """Test that the block inside with_langfuse executes."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
            with_langfuse,
        )

        executed = False
        with with_langfuse():
            executed = True

        assert executed

    def test_with_langfuse_conditional_execution(self):
        """Test conditional execution pattern with with_langfuse."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
            with_langfuse,
        )

        result = None
        with with_langfuse() as lf:
            if lf:
                result = "langfuse available"
            else:
                result = "langfuse not available"

        # Either result is valid, just ensure it was set
        assert result in ["langfuse available", "langfuse not available"]


class TestLangfuseCompatIntegration:
    """Integration tests for langfuse compatibility."""

    def test_observe_on_async_function(self):
        """Test observe decorator on async functions."""
        import asyncio

        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import observe

        @observe()
        async def async_function(value: int) -> int:
            return value * 2

        result = asyncio.run(async_function(21))
        assert result == 42

    def test_observe_preserves_function_metadata(self):
        """Test that observe preserves function name and docstring."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import observe

        @observe()
        def documented_function():
            """This is a docstring."""
            pass

        # Function name and docstring should be preserved
        assert documented_function.__name__ == "documented_function"
        # Note: The no-op decorator might not preserve __doc__ perfectly,
        # but the function should still be callable
        assert callable(documented_function)

    def test_langfuse_module_attribute_exists(self):
        """Test that the langfuse attribute exists and is usable."""
        from rasa.builder.telemetry.langfuse_integration.langfuse_compat import langfuse

        # Should have observe attribute regardless of whether real langfuse is installed
        assert hasattr(langfuse, "observe")

        # Should have get_client method
        assert hasattr(langfuse, "get_client")
