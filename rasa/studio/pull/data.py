import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Text

from rasa.shared.core.flows import Flow
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader, YamlFlowsWriter
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import read_yaml
from rasa.studio.constants import STUDIO_NLU_FILENAME
from rasa.utils.mapper import RasaPrimitiveStorageMapper

logger = logging.getLogger(__name__)

STUDIO_FLOWS_DIR_NAME = "flows"


def merge_nlu_in_directory(
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
    data_path: Path,
    mapper: RasaPrimitiveStorageMapper,
) -> None:
    """
    Merges NLU data by checking for an existing NLU file in the directory.
    If it exists, the new Studio data is merged with the local data.

    Args:
        data_from_studio: The TrainingDataImporter instance for Studio data.
        data_local: The TrainingDataImporter instance for local data.
        data_path: The path to the training data directory.
        mapper: The RasaPrimitiveStorageMapper instance for mapping.
    """
    from rasa.studio.download import pretty_write_nlu_yaml

    nlu_data = data_from_studio.get_nlu_data()
    nlu_file_path = get_nlu_path(data_path, data_local, mapper)

    if nlu_file_path.exists():
        local_nlu = TrainingDataImporter.load_from_dict(
            training_data_paths=[str(nlu_file_path)]
        )
        nlu_data = nlu_data.merge(local_nlu.get_nlu_data())

    if nlu_yaml := nlu_data.nlu_as_yaml():
        pretty_write_nlu_yaml(read_yaml(nlu_yaml), nlu_file_path)


def get_nlu_path(
    base_path: Path,
    data_local: TrainingDataImporter,
    mapper: RasaPrimitiveStorageMapper,
) -> Path:
    """Determines where NLU data should be stored.

    Args:
        base_path: The base path for the training data.
        data_local: The TrainingDataImporter instance for local data.
        mapper: The RasaPrimitiveStorageMapper instance for mapping.

    Returns:
        The path where NLU data should be stored.
    """
    nlu_paths = set()
    for intent in data_local.get_nlu_data().intents:
        for p in mapper.get_file(intent, "intents").get("training", []):
            nlu_paths.add(p)

    return _select_path(nlu_paths, "nlu", base_path, STUDIO_NLU_FILENAME)


def merge_flows_in_directory(
    data_from_studio: TrainingDataImporter,
    data_path: Path,
    mapper: RasaPrimitiveStorageMapper,
) -> None:
    """
    Merges flows data by updating local flow files in a directory with any changes
    from Studio, and then dumping any new flows that do not exist in any local file.

    Args:
        data_from_studio: Training data importer containing the flows from Studio.
        data_path: The path to the directory where local flows reside.
        mapper: Utility for mapping flow IDs to their respective file paths.
    """
    # Extract flows from Studio data and map flow IDs to their instances.
    studio_flows = data_from_studio.get_user_flows()
    studio_flow_map: Dict[Text, Flow] = {
        flow.id: flow for flow in studio_flows.underlying_flows
    }

    # Load existing local flows from the specified directory.
    local_flows_file = TrainingDataImporter.load_from_dict(
        training_data_paths=[str(data_path)]
    )
    local_flows = local_flows_file.get_user_flows().underlying_flows

    # Gather the unique file paths where each local flow is stored.
    local_flow_paths: Set[Path] = _get_local_flow_paths(local_flows, mapper)

    # Track updated flows and update local files with Studio flow data.
    all_updated_flows_ids: List[Text] = []
    for flow_file_path in local_flow_paths:
        updated_flows_ids = _update_flow_file(flow_file_path, studio_flow_map)
        all_updated_flows_ids.extend(updated_flows_ids)

    # Identify new Studio flows and save them as separate files in the directory.
    new_flows = [
        flow
        for flow_id, flow in studio_flow_map.items()
        if flow_id not in all_updated_flows_ids
    ]
    _dump_flows_as_separate_files(new_flows, data_path)


def _get_local_flow_paths(
    local_flows: List[Any],
    mapper: RasaPrimitiveStorageMapper,
) -> Set[Path]:
    """
    Args:
        local_flows: List of local flows.
        mapper: The RasaPrimitiveStorageMapper instance for mapping.

    Returns:
        A set of paths for the local flow files.
    """
    paths: Set[Path] = set()
    for flow in local_flows:
        paths.update(mapper.get_file(flow.id, "flows").get("training", []))
    return paths


def _update_flow_file(
    flow_file_path: Path, studio_flows_map: Dict[Text, Any]
) -> List[Text]:
    """
    Reads a flow file, updates outdated flows, and replaces them with studio versions.

    Args:
        flow_file_path: The path to the flow file.
        studio_flows_map: A dictionary mapping flow IDs to their updated versions.

    Returns:
        A list of Flows IDs from the updated flow file.
    """
    file_flows = YAMLFlowsReader.read_from_file(flow_file_path, False)

    # Build a list of flows, replacing any outdated flow with its studio version
    updated_flows = [
        studio_flows_map.get(flow.id, flow) or flow
        for flow in file_flows.underlying_flows
    ]

    # If the updated flows differ from the original file flows, write them back
    if updated_flows != file_flows.underlying_flows:
        YamlFlowsWriter.dump(
            flows=updated_flows,
            filename=flow_file_path,
            should_clean_json=True,
        )

    return [flow.id for flow in updated_flows]


def _dump_flows_as_separate_files(flows: List[Any], data_path: Path) -> None:
    """Dump flow into separate files within a directory.

    Creates a new directory under the given data_path and writes each flow
    into a separate YAML file. Each file is named after the flow's id.

    Args:
        flows: List of new flows to be dumped.
        data_path: The path to the directory where the files will be created.
    """
    # If there are no flows, don't create a directory.
    if not flows:
        return

    new_flows_dir = data_path / STUDIO_FLOWS_DIR_NAME
    new_flows_dir.mkdir(parents=True, exist_ok=True)
    for flow in flows:
        file_name = f"{flow.id}.yml"
        file_path = new_flows_dir / file_name
        single_flow_list = FlowsList(underlying_flows=[flow])
        YamlFlowsWriter.dump(
            flows=single_flow_list.underlying_flows,
            filename=file_path,
            should_clean_json=True,
        )


def _select_path(
    paths: Set[Path], primitive_type: str, default_path: Path, default: str
) -> Path:
    """Selects a path from a set of paths.

    If exactly one path exists, returns it.
    If multiple exist, returns one with a warning.
    If none exist, returns a default path.

    Args:
        paths: A set of paths.
        primitive_type: The type of the primitive (e.g., "domain", "nlu").
        default_path: The default path to use if no paths exist.
        default: The default file name.

    Returns:
        The selected path.
    """
    if len(paths) == 1:
        path = paths.pop()
    elif len(paths) > 1:
        path = paths.pop()
        logger.warning(
            f"Saving {primitive_type} to {path}. "
            f"Please keep Studio-related {primitive_type} in a single file."
        )
    else:
        path = default_path / Path(default)
        logger.info(f"Saving {primitive_type} to {path}.")
    return path
