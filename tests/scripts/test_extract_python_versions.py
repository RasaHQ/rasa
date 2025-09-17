"""Tests for extract_python_versions.py script."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.extract_python_versions import (
    extract_python_versions_from_pyproject,
    generate_version_list,
    parse_version_constraint,
)


class TestParseVersionConstraint:
    """Test parse_version_constraint function."""

    def test_parse_version_constraint_with_quotes(self):
        """Test parsing version constraint with quotes."""
        min_ver, max_ver = parse_version_constraint('">=3.9.2,<3.12"')
        assert min_ver == "3.9"
        assert max_ver == "3.12"

    def test_parse_version_constraint_without_quotes(self):
        """Test parsing version constraint without quotes."""
        min_ver, max_ver = parse_version_constraint(">=3.9.2,<3.12")
        assert min_ver == "3.9"
        assert max_ver == "3.12"

    def test_parse_version_constraint_with_spaces(self):
        """Test parsing version constraint with spaces."""
        min_ver, max_ver = parse_version_constraint(" >=3.9.2, <3.12 ")
        assert min_ver == "3.9"
        assert max_ver == "3.12"

    def test_parse_version_constraint_min_only(self):
        """Test parsing version constraint with only minimum version."""
        min_ver, max_ver = parse_version_constraint(">=3.9.2")
        assert min_ver == "3.9"
        assert max_ver is None

    def test_parse_version_constraint_max_only(self):
        """Test parsing version constraint with only maximum version."""
        min_ver, max_ver = parse_version_constraint("<3.12")
        assert min_ver is None
        assert max_ver == "3.12"

    def test_parse_version_constraint_no_constraints(self):
        """Test parsing version constraint with no constraints."""
        min_ver, max_ver = parse_version_constraint("3.9.2")
        assert min_ver is None
        assert max_ver is None

    def test_parse_version_constraint_empty_string(self):
        """Test parsing empty version constraint."""
        min_ver, max_ver = parse_version_constraint("")
        assert min_ver is None
        assert max_ver is None


class TestGenerateVersionList:
    """Test generate_version_list function."""

    def test_generate_version_list_same_major(self):
        """Test generating version list for same major version."""
        versions = generate_version_list("3.9", "3.12")
        expected = ["3.9", "3.10", "3.11"]
        assert versions == expected

    def test_generate_version_list_different_majors(self):
        """Test generating version list for different major versions."""
        versions = generate_version_list("3.8", "4.2")
        expected = [
            "3.8",
            "3.9",  # 3.x versions from 3.8
            "4.0",
            "4.1",  # 4.x versions up to 4.1
        ]
        assert versions == expected

    def test_generate_version_list_single_major(self):
        """Test generating version list for single major version."""
        versions = generate_version_list("3.9", "3.10")
        expected = ["3.9"]
        assert versions == expected

    def test_generate_version_list_none_min(self):
        """Test generating version list with None min version."""
        versions = generate_version_list(None, "3.12")
        assert versions == []

    def test_generate_version_list_none_max(self):
        """Test generating version list with None max version."""
        versions = generate_version_list("3.9", None)
        assert versions == []

    def test_generate_version_list_both_none(self):
        """Test generating version list with both None."""
        versions = generate_version_list(None, None)
        assert versions == []


class TestExtractPythonVersionsFromPyproject:
    """Test extract_python_versions_from_pyproject function."""

    def test_extract_versions_valid_pyproject(self):
        """Test extracting versions from valid pyproject.toml."""
        pyproject_content = """
[tool.poetry.dependencies]
python = ">=3.9.2,<3.12"
other_dependency = "1.0.0"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(pyproject_content)
            f.flush()

            try:
                versions = extract_python_versions_from_pyproject(Path(f.name))
                expected = ["3.9", "3.10", "3.11"]
                assert versions == expected
            finally:
                Path(f.name).unlink()

    def test_extract_versions_with_quotes(self):
        """Test extracting versions with quoted version constraint."""
        pyproject_content = """
[tool.poetry.dependencies]
python = '>=3.8.0,<3.10'
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(pyproject_content)
            f.flush()

            try:
                versions = extract_python_versions_from_pyproject(Path(f.name))
                expected = ["3.8", "3.9"]
                assert versions == expected
            finally:
                Path(f.name).unlink()

    def test_extract_versions_file_not_found(self):
        """Test extracting versions from non-existent file."""
        with pytest.raises(FileNotFoundError):
            extract_python_versions_from_pyproject(Path("nonexistent.toml"))

    def test_extract_versions_no_python_constraint(self):
        """Test extracting versions when no Python constraint is found."""
        pyproject_content = """
[tool.poetry.dependencies]
other_dependency = "1.0.0"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(pyproject_content)
            f.flush()

            try:
                with pytest.raises(
                    ValueError, match="Python version constraint not found"
                ):
                    extract_python_versions_from_pyproject(Path(f.name))
            finally:
                Path(f.name).unlink()

    def test_extract_versions_invalid_constraint(self):
        """Test extracting versions with invalid constraint format."""
        pyproject_content = """
[tool.poetry.dependencies]
python = "3.9.2"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(pyproject_content)
            f.flush()

            try:
                with pytest.raises(
                    ValueError, match="Could not parse version constraint"
                ):
                    extract_python_versions_from_pyproject(Path(f.name))
            finally:
                Path(f.name).unlink()

    def test_extract_versions_multiple_sections(self):
        """Test extracting versions when constraints appear in multiple sections."""
        pyproject_content = """
[tool.poetry.dependencies]
python = ">=3.9.2,<3.12"
other_dependency = "1.0.0"

[tool.poetry.group.dev.dependencies]
python = ">=3.8.0,<3.10"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(pyproject_content)
            f.flush()

            try:
                versions = extract_python_versions_from_pyproject(Path(f.name))
                # Should use the first occurrence
                expected = ["3.9", "3.10", "3.11"]
                assert versions == expected
            finally:
                Path(f.name).unlink()


class TestMainFunction:
    """Test main function and CLI interface."""

    def test_main_with_default_args(self):
        """Test main function with default arguments."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[tool.poetry.dependencies]
python = ">=3.9.2,<3.12"
""")
            f.flush()

            try:
                with patch(
                    "sys.argv",
                    ["extract_python_versions.py", "--pyproject-path", f.name],
                ):
                    from scripts.extract_python_versions import main

                    # This should not raise an exception
                    main()
            finally:
                Path(f.name).unlink()

    def test_main_file_not_found(self):
        """Test main function with non-existent file."""
        with patch(
            "sys.argv",
            ["extract_python_versions.py", "--pyproject-path", "nonexistent.toml"],
        ):
            with patch("sys.exit") as mock_exit:
                from scripts.extract_python_versions import main

                main()
                mock_exit.assert_called_once_with(1)

    def test_main_comma_separated_output(self):
        """Test main function outputs comma-separated format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[tool.poetry.dependencies]
python = ">=3.9.2,<3.12"
""")
            f.flush()

            try:
                with patch(
                    "sys.argv",
                    ["extract_python_versions.py", "--pyproject-path", f.name],
                ):
                    with patch("builtins.print") as mock_print:
                        from scripts.extract_python_versions import main

                        main()
                        mock_print.assert_called_once_with("3.9,3.10,3.11")
            finally:
                Path(f.name).unlink()
