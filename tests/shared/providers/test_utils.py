from rasa.shared.providers._configs.utils import resolve_aliases


def test_resolve_aliases_with_single_alias() -> None:
    config = {"old_key": "value1", "another_key": "value2"}
    alias_mapping = {"old_key": "new_key"}

    result = resolve_aliases(config, alias_mapping)

    assert result == {"new_key": "value1", "another_key": "value2"}
    assert "old_key" not in result


def test_resolve_aliases_with_multiple_aliases() -> None:
    config = {"old_key1": "value1", "old_key2": "value2"}
    alias_mapping = {"old_key1": "new_key1", "old_key2": "new_key2"}

    result = resolve_aliases(config, alias_mapping)

    assert result == {"new_key1": "value1", "new_key2": "value2"}
    assert "old_key1" not in result
    assert "old_key2" not in result


def test_resolve_aliases_with_no_aliases() -> None:
    config = {"key1": "value1", "key2": "value2"}
    alias_mapping = {"non_existent_key": "new_key"}

    result = resolve_aliases(config, alias_mapping)

    assert result == config


def test_resolve_aliases_with_conflicting_keys() -> None:
    config = {"old_key": "value1", "new_key": "value2"}
    alias_mapping = {"old_key": "new_key"}

    result = resolve_aliases(config, alias_mapping)

    assert result == {"new_key": "value1"}
    assert result["new_key"] == "value1"
    assert "old_key" not in result


def test_resolve_aliases_with_empty_config() -> None:
    config = {}
    alias_mapping = {"old_key": "new_key"}

    result = resolve_aliases(config, alias_mapping)

    assert result == {}
