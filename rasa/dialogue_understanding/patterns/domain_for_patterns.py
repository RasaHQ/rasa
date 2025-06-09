from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Set, Text

import importlib_resources
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from rasa.dialogue_understanding.stack.frames.pattern_frame import PatternFlowStackFrame
from rasa.shared.core.constants import (
    DEFAULT_ACTION_NAMES,
    DEFAULT_INTENTS,
    RULE_SNIPPET_ACTION_NAME,
)
from rasa.shared.core.slots import AnySlot, BooleanSlot, CategoricalSlot, TextSlot

ACTIONS: List[Text] = [
    *DEFAULT_ACTION_NAMES,
    "validate_{{context.collect}}",
]
EXCLUDED_ACTIONS: Set[Text] = {RULE_SNIPPET_ACTION_NAME}

INTENTS: List[Text] = [*DEFAULT_INTENTS]

CONTEXT_FIELD_TYPES: Dict[Text, Text] = {
    "canceled_name": CategoricalSlot.type_name,
    "names": CategoricalSlot.type_name,
    "collect": CategoricalSlot.type_name,
    "utter": CategoricalSlot.type_name,
    "collect_action": CategoricalSlot.type_name,
    "previous_flow_name": CategoricalSlot.type_name,
    "reset_flow_id": CategoricalSlot.type_name,
    "is_reset_only": BooleanSlot.type_name,
    "canceled_frames": AnySlot.type_name,
    "corrected_slots": AnySlot.type_name,
    "rejections": AnySlot.type_name,
    "info": AnySlot.type_name,
    "reason": TextSlot.type_name,
    "error_type": TextSlot.type_name,
}

FLOW_NAME_VALUES = "FLOW_NAME"
FLOW_ID_VALUES = "FLOW_ID"
SLOT_NAME_VALUES = "SLOT_NAME"
RESPONSE_NAME_VALUES = "RESPONSE_NAME"
ACTION_NAME_VALUES = "ACTION_NAME"

CONTEXT_FIELD_VALUES: Dict[Text, Text] = {
    "canceled_name": FLOW_NAME_VALUES,
    "names": FLOW_NAME_VALUES,
    "previous_flow_name": FLOW_NAME_VALUES,
    "reset_flow_id": FLOW_ID_VALUES,
    "collect": SLOT_NAME_VALUES,
    "utter": RESPONSE_NAME_VALUES,
    "collect_action": ACTION_NAME_VALUES,
}

PATTERNS_MODULE_BASE = "rasa.dialogue_understanding.patterns"


class ContextField(BaseModel):
    """Element in the `contexts` mapping of the domain."""

    patterns: List[Text] = Field(
        ..., description="Patterns that reference this context field."
    )
    type: Text = Field(
        ..., description="Slot type (categorical, text, boolean, any …)."
    )
    values: Optional[List[Text]] = Field(
        None,
        description="Optional placeholder that restricts which values a slot can take "
        "(FLOW_NAME, SLOT_NAME, …).",
    )

    @field_validator("type")
    def _validate_slot_type(cls, v: str) -> str:
        allowed_types = list(set(CONTEXT_FIELD_TYPES.values()))
        if v not in allowed_types:
            raise ValueError(
                f"Unsupported type '{v}'. "
                f"Must be one of: {', '.join(allowed_types)}."
            )
        return v

    @field_validator("values")
    def _validate_values_placeholder(
        cls, v: Optional[str], values: ValidationInfo
    ) -> Optional[str]:
        if v is None:
            return v

        allowed_values = set(CONTEXT_FIELD_VALUES.values())
        if not set(v).issubset(allowed_values):
            raise ValueError(
                f"Unsupported values placeholder '{v}'. "
                f"Must be one of {', '.join(allowed_values)}."
            )

        slot_type = values.data.get("type")
        if slot_type != CategoricalSlot.type_name:
            raise ValueError(
                "`values` can only be specified for categorical slots "
                f"(got slot type '{slot_type}')."
            )

        return v


class PatternDomain(BaseModel):
    """Complete domain that is generated for the default patterns."""

    actions: List[Text]
    intents: List[Text]
    contexts: Dict[Text, ContextField]


def build_contexts_from_patterns() -> Dict[str, Dict[str, Any]]:
    """Builds a dictionary of contexts from the pattern classes.

    Returns:
        A dictionary where each key is a field name and the value is a dictionary
    """
    patterns_folder = str(importlib_resources.files(PATTERNS_MODULE_BASE))
    contexts_map: Dict[str, Dict[str, Any]] = {}

    # Dynamically gather all .py files
    for root, _, files in os.walk(patterns_folder):
        for file in files:
            if not file.endswith(".py"):
                continue

            try:
                module_name = f"{PATTERNS_MODULE_BASE}.{file[:-3]}"
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            # Inspect classes in that module
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if not is_dataclass(cls):
                    continue

                # The cls has to be a subclass of PatternFlowStackFrame
                if cls == PatternFlowStackFrame or not issubclass(
                    cls, PatternFlowStackFrame
                ):
                    continue

                for f in fields(cls):
                    field_name = f.name
                    if field_name not in contexts_map:
                        field_type = CONTEXT_FIELD_TYPES.get(
                            field_name, TextSlot.type_name
                        )
                        contexts_map[field_name] = {
                            "patterns": set(),
                            "type": field_type,
                        }
                        values: Optional[Text] = CONTEXT_FIELD_VALUES.get(field_name)
                        if values:
                            contexts_map[field_name]["values"] = [values]

                    # Add the pattern name to the set
                    pattern_name = cls.type()
                    contexts_map[field_name]["patterns"].add(pattern_name)

    # Convert "patterns" from set to list, for a clean final structure
    for field_name, details in contexts_map.items():
        details["patterns"] = sorted(list(details["patterns"]))

    return contexts_map


def generate_domain_for_default_patterns() -> PatternDomain:
    """
    Generate the domain for pattern-based flows as a strongly-typed object.

    Returns: PatternDomain Pydantic model containing actions, intents and contexts.
    """
    actions = [action for action in ACTIONS if action not in EXCLUDED_ACTIONS]
    raw_contexts = build_contexts_from_patterns()
    contexts = {
        field_name: ContextField(**details)
        for field_name, details in raw_contexts.items()
    }
    return PatternDomain(actions=actions, intents=INTENTS, contexts=contexts)
