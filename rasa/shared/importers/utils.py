from typing import Any, Dict, Iterable, List, Optional, Text

from pydantic import BaseModel, Field

from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.yaml_flows_io import KEY_FLOWS, get_flows_as_json
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.json_utils import extract_values


class CALMUserData(BaseModel):
    """All pieces that will be uploaded to Rasa Studio."""

    flows: Dict[str, Any] = Field(default_factory=dict)
    domain: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    endpoints: Dict[str, Any] = Field(default_factory=dict)
    nlu: Dict[str, Any] = Field(default_factory=dict)


DOMAIN_KEYS = [
    "version",
    "actions",
    "responses",
    "slots",
    "intents",
    "entities",
    "forms",
    "session_config",
]


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    from rasa.shared.nlu.training_data import loading

    training_data_sets = [loading.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingData().merge(*training_data_sets)


def story_graph_from_paths(
    files: List[Text], domain: Domain, exclusion_percentage: Optional[int] = None
) -> StoryGraph:
    """Returns the `StoryGraph` from paths."""
    from rasa.shared.core.training_data import loading

    story_steps = loading.load_data_from_files(files, domain, exclusion_percentage)
    return StoryGraph(story_steps)


def flows_from_paths(files: List[Text], domain: Optional[Domain] = None) -> FlowsList:
    """Returns the flows from paths.

    Args:
        files: List of flow file paths to load.
        domain: Optional domain for validation. If provided, exit_if conditions
               will be validated against defined slots.
    """
    from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader

    flows = FlowsList(underlying_flows=[])
    for file in files:
        flows = flows.merge(
            YAMLFlowsReader.read_from_file(file), ignore_duplicates=False
        )
    flows.validate(domain)
    return flows


def extract_calm_import_parts_from_importer(
    importer: TrainingDataImporter,
    config: Optional[Dict[str, Any]] = None,
    endpoints: Optional[Dict[str, Any]] = None,
) -> CALMUserData:
    """Extracts CALMUserData from a TrainingDataImporter.

    Args:
        importer: The training data importer
        data_paths: The path(s) to the training data for flows
        config: Optional config dict, if not provided will use importer.get_config()
        endpoints: Optional endpoints dict, defaults to empty dict

    Returns:
        CALMUserData containing flows, domain, config, endpoints, and nlu data
    """
    # Extract config
    if config is None:
        config = importer.get_config()

    # Extract domain
    domain_from_files = importer.get_user_domain().as_dict()
    domain = extract_values(domain_from_files, DOMAIN_KEYS)

    # Extract flows
    flows = importer.get_user_flows()
    flows_dict = {KEY_FLOWS: get_flows_as_json(flows)}

    # Extract NLU data
    nlu_data = importer.get_nlu_data()
    nlu_examples = nlu_data.filter_training_examples(
        lambda ex: ex.get("intent") in nlu_data.intents
    )
    nlu_dict = RasaYAMLWriter().training_data_to_dict(nlu_examples)

    # Use provided endpoints or default to empty dict
    if endpoints is None:
        endpoints = {}

    return CALMUserData(
        flows=flows_dict or {},
        domain=domain or {},
        config=config or {},
        endpoints=endpoints or {},
        nlu=nlu_dict or {},
    )
