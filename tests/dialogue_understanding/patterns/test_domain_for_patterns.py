import json
from pathlib import Path
from typing import Dict, List, Set, Text

import pytest

from rasa.dialogue_understanding.patterns.domain_for_patterns import (
    ACTIONS,
    CONTEXT_FIELD_TYPES,
    CONTEXT_FIELD_VALUES,
    EXCLUDED_ACTIONS,
    FLOW_NAME_VALUES,
    INTENTS,
    ContextField,
    PatternDomain,
    build_contexts_from_patterns,
    generate_domain_for_default_patterns,
)
from rasa.shared.core.slots import (
    CategoricalSlot,
    TextSlot,
)

CONTEXTS_SNAPSHOT_PATH = Path(__file__).with_name(
    "build_contexts_from_patterns_snapshot.json"
)


def test_context_field_accepts_supported_slot_type():
    cf = ContextField(
        patterns=["SomePattern"],
        type=CategoricalSlot.type_name,
        values=[FLOW_NAME_VALUES],
    )
    assert cf.type == CategoricalSlot.type_name
    assert cf.values == [FLOW_NAME_VALUES]


def test_context_field_rejects_unknown_slot_type():
    with pytest.raises(ValueError, match="Unsupported type"):
        ContextField(patterns=["P"], type="imaginary", values=None)


def test_context_field_rejects_values_for_non_categorical_slot():
    with pytest.raises(ValueError, match="only be specified for categorical"):
        ContextField(
            patterns=["OtherPattern"],
            type=TextSlot.type_name,
            values=[FLOW_NAME_VALUES],
        )


def test_context_field_rejects_unknown_placeholder():
    wrong_placeholder = "UNKNOWN_PLACEHOLDER"
    assert wrong_placeholder not in CONTEXT_FIELD_VALUES.values()

    with pytest.raises(ValueError, match="Unsupported values placeholder"):
        ContextField(
            patterns=["OtherPattern"],
            type=CategoricalSlot.type_name,
            values=[wrong_placeholder],
        )


def test_build_contexts_from_patterns_produces_expected_structure():
    contexts = build_contexts_from_patterns()

    # Must return a non-empty mapping
    assert isinstance(contexts, dict) and contexts

    # Every entry complies with declared slot-type mapping
    allowed_types = set(CONTEXT_FIELD_TYPES.values())
    for details in contexts.values():
        assert set(details) >= {"patterns", "type"}
        assert isinstance(details["patterns"], list) and details["patterns"]
        assert details["type"] in allowed_types


def test_generate_domain_for_default_patterns_returns_valid_domain():
    domain = generate_domain_for_default_patterns()
    assert isinstance(domain, PatternDomain)

    for action in ACTIONS:
        if action not in EXCLUDED_ACTIONS:
            assert action in domain.actions

    for action in EXCLUDED_ACTIONS:
        assert action not in domain.actions

    assert len(domain.actions) == len(set(domain.actions))

    for intent in INTENTS:
        assert intent in domain.intents

    for name, context in domain.contexts.items():
        assert isinstance(context, ContextField)
        assert context.type in CONTEXT_FIELD_TYPES.values()
        assert context.patterns


def test_all_context_fields_are_mapped():
    """Guard test that makes sure:
        - No new context field slips in unnoticed.
        - If a new `categorical` field is added, the developer also adds the
          required entries to CONTEXT_FIELD_TYPES and CONTEXT_FIELD_VALUES.

    When this test fails:
        1. If the new context field is `text` (default), add its name to the
        snapshot JSON.
        2. If it is `categorical`, first update the two mapping dictionaries
        mentioned above, then refresh the snapshot.
    """

    def _read_snapshot() -> Dict[Text, Dict]:
        """Load the JSON snapshot that lives next to this test‐file.

        Raises:
            AssertionError: If the snapshot file is missing.
        """
        if not CONTEXTS_SNAPSHOT_PATH.is_file():
            raise AssertionError(
                f"Snapshot file '{CONTEXTS_SNAPSHOT_PATH}' not found. "
                "Create or refresh it with the helper script in this folder."
            )

        return json.loads(CONTEXTS_SNAPSHOT_PATH.read_text())

    def _human_join(items: List[Text]) -> Text:
        """Return a human-readable, comma-separated Texting."""
        return ", ".join(sorted(items))

    expected: Dict[Text, Dict] = _read_snapshot()
    current: Dict[Text, Dict] = build_contexts_from_patterns()

    expected_fields: Set[Text] = set(expected)
    current_fields: Set[Text] = set(current)

    new_fields = current_fields - expected_fields
    removed_fields = expected_fields - current_fields

    problems: List[Text] = []
    if new_fields:
        problems.append(
            f"New context field(s) detected: {_human_join(list(new_fields))}"
        )

        for field in new_fields:
            slot_type = current[field]["type"]
            if slot_type == CategoricalSlot.type_name:
                # categorical slots must be present in both mapping dicts
                missing_mappings = []
                if field not in CONTEXT_FIELD_TYPES:
                    missing_mappings.append("CONTEXT_FIELD_TYPES")
                if field not in CONTEXT_FIELD_VALUES:
                    missing_mappings.append("CONTEXT_FIELD_VALUES")
                if missing_mappings:
                    problems.append(
                        f"  - '{field}' is categorical but missing mapping(s): "
                        f"{_human_join(missing_mappings)}"
                    )
            else:
                # text slots do not need explicit mappings; developer only has to
                # update the snapshot.
                problems.append(
                    f"  - '{field}' looks like a text slot. "
                    "If that is intended, update the snapshot JSON."
                )

    if removed_fields:
        problems.append(
            f"Context field(s) disappeared: {_human_join(list(removed_fields))}. "
            "If removal was intentional, also delete them from the snapshot."
        )

    if problems:
        raise AssertionError(
            "\n".join(
                [
                    "The set of context fields has changed.",
                    *problems,
                    "",
                    f"Update the snapshot file at '{CONTEXTS_SNAPSHOT_PATH}' once the "
                    "changes are intentional and the mapping dictionaries are "
                    "in sync.  A helper script is provided next to this test:",
                    "    python update_context_field_snapshot.py",
                ]
            )
        )
