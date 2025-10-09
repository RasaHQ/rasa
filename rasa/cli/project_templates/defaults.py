from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Text

from pydantic import BaseModel, ConfigDict, Field

from rasa.dialogue_understanding.patterns.domain_for_patterns import (
    ContextField,
    generate_domain_for_default_patterns,
)
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import FlowSyncImporter, TrainingDataImporter
from rasa.shared.utils.llm import SystemPrompts, get_system_default_prompts
from rasa.shared.utils.yaml import read_yaml, write_yaml


class Response(BaseModel):
    text: str = Field(..., description="The human-readable response string.")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata used by Rasa at runtime."
    )


class Slot(BaseModel):
    name: Text = Field(..., description="Name of the slot.")
    type: Text = Field(..., description="Type of the slot.")
    mappings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Mappings for the slot."
    )
    model_config = ConfigDict(extra="allow")


class PatternFlow(BaseModel):
    description: str = Field(..., description="Description of the flow.")
    name: Text = Field(..., description="Name of the flow.")
    steps: List[Any] = Field(..., description="List of steps in the flow.")
    nlu_trigger: Optional[Any] = Field(None, description="NLU trigger for the flow.")
    persisted_slots: Optional[List[Text]] = Field(None, description="Slots to persist.")


class PatternDefaults(BaseModel):
    """Content of `default_flows_for_patterns.yml` after it is loaded from disk."""

    responses: Dict[Text, List[Response]] = Field(
        default_factory=dict,
        description="Response templates available out-of-the-box.",
    )
    slots: Dict[Text, Slot] = Field(
        default_factory=dict, description="Slot definitions used by the default flows."
    )
    flows: Dict[Text, PatternFlow] = Field(
        default_factory=dict, description="Default flows (conversation patterns)."
    )


class RasaDefaults(PatternDefaults):
    """Full set of defaults that Rasa injects when a project is created."""

    prompts: SystemPrompts = Field(..., description="Built-in system prompts.")
    actions: List[Text] = Field(..., description="Default actions shipped by Rasa.")
    intents: List[Text] = Field(..., description="Default intents shipped by Rasa.")
    contexts: Dict[Text, ContextField] = Field(
        ..., description="Context fields that can appear on the conversation stack."
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


def _get_domain_from_importer(config: Dict[Text, Any]) -> Domain:
    """Get the domain from the TrainingDataImporter.

    Args:
        config: The config.yml file data.

    Returns:
        A Domain object .
    """
    # We create the file with delete=False so that it can be re-opened on
    # Windows. If delete=True, the file is opened with the _O_TEMPORARY flag
    # which blocks any second open() call.
    tmp = tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False)

    try:
        path = tmp.name

        # write_yaml() re-opens the same path. On Windows an already-open
        # handle keeps the file locked for further opens, so we close
        # the first handle before we call write_yaml().
        tmp.close()
        write_yaml(config, path)

        importer = TrainingDataImporter.load_from_config(
            domain_path=FlowSyncImporter.default_pattern_path(),
            config_path=path,
        )
        return importer.get_domain()

    finally:
        # Because we passed delete=False above, Python will not clean the file
        # To avoid leaving garbage in the temp directory after the tests run,
        # we remove it explicitly.
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def get_pattern_defaults(config: Dict[Text, Any]) -> PatternDefaults:
    """Get the default flows, responses and slots for patterns.

    Args:
        config: The config.yml file data.

    Returns:
        A PatternDefaults object containing the defaults.
    """
    domain = _get_domain_from_importer(config)
    domain_dict = domain.as_dict()

    # Make sure all the slots are exported, including the builtin slots
    domain_dict["slots"] = {slot.name: slot.to_dict() for slot in domain.slots}

    return PatternDefaults(**domain_dict)


def get_rasa_defaults(config_yaml: Text, endpoints_yaml: Text) -> RasaDefaults:
    """Get the default values for a Rasa project.

    Args:
        config_yaml: The content of the config.yml file.
        endpoints_yaml: The content of the endpoints.yml file.

    Returns:
        A RasaDefaults object containing the default values for the project.
    """
    config = read_yaml(config_yaml)
    # Does not read from file, it converts the YAML content to a dictionary.
    endpoints = read_yaml(endpoints_yaml)

    prompts = get_system_default_prompts(config, endpoints)
    pattern_domain = generate_domain_for_default_patterns()
    pattern_defaults = get_pattern_defaults(config)

    return RasaDefaults(
        prompts=prompts,
        actions=pattern_domain.actions,
        intents=pattern_domain.intents,
        contexts=pattern_domain.contexts,
        responses=pattern_defaults.responses,
        slots=pattern_defaults.slots,
        flows=pattern_defaults.flows,
    )
