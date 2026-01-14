"""Tests for the patched pypred module."""

from unittest.mock import patch

from rasa.utils.pypred import Predicate


def test_pypred_patched_predicate_still_works():
    """Test that the patched pypred still works."""
    # Create a predicate in the read-only directory
    predicate = Predicate("a > 5")
    # This should not raise any errors about being unable to write files
    assert predicate.evaluate({"a": 6}) is True
    assert predicate.evaluate({"a": 4}) is False


def test_pypred_patch_disables_write_tables():
    """Test that the patched yacc function disables write_tables."""
    with patch("rasa.utils.pypred._original_yacc") as mock_yacc:
        from rasa.utils.pypred import patched_yacc

        # Call the patched function
        patched_yacc()

        # Verify that write_tables was set to False
        mock_yacc.assert_called_once()
        _, kwargs = mock_yacc.call_args
        assert kwargs["write_tables"] is False
