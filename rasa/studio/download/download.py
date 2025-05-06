import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import questionary
import structlog
from ruamel import yaml
from ruamel.yaml.scalarstring import LiteralScalarString

import rasa.cli.utils
import rasa.shared.utils.cli
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DOMAIN_PATHS,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import read_yaml
from rasa.studio import data_handler
from rasa.studio.config import StudioConfig
from rasa.studio.constants import (
    STUDIO_DOMAIN_FILENAME,
    STUDIO_FLOWS_FILENAME,
    STUDIO_NLU_FILENAME,
)
from rasa.studio.data_handler import StudioDataHandler, import_data_from_studio
from rasa.studio.download.domains import merge_domain_with_overwrite
from rasa.studio.download.flows import merge_flows_with_overwrite
from rasa.utils.mapper import RasaPrimitiveStorageMapper

structlogger = structlog.getLogger(__name__)


def handle_download(args: argparse.Namespace) -> None:
    """Main function to handle downloading and merging data.

    Args:
        args: The command line arguments.
    """
    handler = StudioDataHandler(
        studio_config=StudioConfig.read_config(),
        assistant_name=args.assistant_name[0],
    )
    handler.request_all_data()

    domain_path, data_paths = _prepare_data_and_domain_paths(args)

    # Handle config and endpoints.
    config_path, write_config = _handle_file_overwrite(
        args.config, DEFAULT_CONFIG_PATH, "config"
    )
    endpoints_path, write_endpoints = _handle_file_overwrite(
        args.endpoints, DEFAULT_ENDPOINTS_PATH, "endpoints"
    )
    message_parts = []
    config_path = config_path if write_config else None
    endpoints_path = endpoints_path if write_endpoints else None

    if config_path:
        config_data = handler.get_config()
        if not config_data:
            rasa.shared.utils.cli.print_error_and_exit("No config data found.")
        with open(config_path, "w") as f:
            f.write(config_data)
            message_parts.append(f"config to '{config_path}'")
    if endpoints_path:
        endpoints_data = handler.get_endpoints()
        if not endpoints_data:
            raise ValueError("No endpoints data found.")
        with open(endpoints_path, "w") as f:
            f.write(endpoints_data)
            message_parts.append(f"endpoints to '{endpoints_path}'")
    if message_parts:
        message = "Downloaded " + " and ".join(message_parts)
        structlogger.info("studio.download.config_endpoints", event_info=message)

    if not args.overwrite:
        _handle_download_no_overwrite(handler, domain_path, data_paths)
    else:
        _handle_download_with_overwrite(handler, domain_path, data_paths)


def _prepare_data_and_domain_paths(args: argparse.Namespace) -> Tuple[Path, List[Path]]:
    """Prepars the domain and data paths based on the provided arguments.

    Args:
        args: The command line arguments.

    Returns:
        A tuple containing the domain path and a list of data paths.
    """
    # Prepare domain path.
    domain_path = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATHS, none_is_valid=True
    )
    domain_or_default_path = args.domain or DEFAULT_DOMAIN_PATH

    if domain_path is None:
        domain_path = Path(domain_or_default_path)
        domain_path.touch()

    if isinstance(domain_path, str):
        domain_path = Path(domain_path)

    if domain_path.is_file():
        if not args.overwrite:
            domain_path.unlink()
            domain_path.touch()

    if domain_path.is_dir():
        if not args.overwrite:
            domain_path = domain_path / STUDIO_DOMAIN_FILENAME
            domain_path.touch()

    # Prepare data paths.
    data_paths: List[Path] = []
    for f in args.data:
        data_path = rasa.cli.utils.get_validated_path(
            f, "data", DEFAULT_DATA_PATH, none_is_valid=True
        )

        if data_path is None:
            data_path = Path(f)
            data_path.mkdir(parents=True, exist_ok=True)
        else:
            data_path = Path(data_path)

        if data_path.is_file() or data_path.is_dir():
            data_paths.append(data_path)
        else:
            data_path.mkdir(parents=True, exist_ok=True)
            data_paths.append(data_path)

    # Remove duplicates while preserving order.
    data_paths = list(dict.fromkeys(data_paths))
    return domain_path, data_paths


def _handle_file_overwrite(
    file_path: Optional[str], default_path: str, file_type: str
) -> Tuple[Optional[Path], bool]:
    """Determines whether to overwrite an existing file or create a new one.

    Works for config and endpoints at this moment.

    Args:
        file_path: The path to the file.
        default_path: The default path to use if no file path is provided.
        file_type: The type of the file (e.g., config, endpoints).

    Returns:
        A tuple of the file path and a boolean indicating overwrite status.
    """
    file_already_exists = rasa.cli.utils.get_validated_path(
        file_path, file_type, default_path, none_is_valid=True
    )
    write_file = False
    path = None
    file_or_default_path = file_path or default_path

    if file_already_exists is None:
        path = Path(file_or_default_path)
        if path.is_dir():
            path = path / default_path
        return path, True

    if questionary.confirm(
        f"{file_type.capitalize()} file '{file_or_default_path}' already exists. "
        f"Do you want to overwrite it?"
    ).ask():
        write_file = True
        path = Path(file_or_default_path)
    return path, write_file


def _handle_download_no_overwrite(
    handler: StudioDataHandler, domain_path: Path, data_paths: List[Path]
) -> None:
    """Handles downloading without overwriting existing files.

    Args:
        handler: The StudioDataHandler instance.
        domain_path: The path to the domain file or directory.
        data_paths: The paths to the data files or directories.
    """
    data_from_studio, data_local = import_data_from_studio(
        handler, domain_path, data_paths
    )
    _merge_domain_no_overwrite(domain_path, data_from_studio, data_local)
    _merge_data_no_overwrite(data_paths, handler, data_from_studio, data_local)


def _merge_domain_no_overwrite(
    domain_path: Path,
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
) -> None:
    """
    Merges the domain data without overwriting.

    If the domain path is a directory, a new domain file is created under that folder.
    If the domain path is a file, it merges both domains into that file.

    Args:
        domain_path: The path to the domain file or directory.
        data_from_studio: The Studio data importer.
        data_local: The local data importer.
    """
    if domain_path.is_dir():
        _merge_directory_domain(domain_path, data_from_studio, data_local)
    elif domain_path.is_file():
        _merge_file_domain(domain_path, data_from_studio, data_local)


def _merge_directory_domain(
    domain_path: Path,
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
) -> None:
    """Merges domain data without overwriting when the domain is a directory.

    Args:
        domain_path: The path to the domain directory.
        data_from_studio: The Studio data importer.
        data_local: The local data importer.
    """
    from rasa.shared.core.domain import Domain

    studio_domain_path = domain_path / STUDIO_NLU_FILENAME
    new_domain_data = data_handler.combine_domains(
        data_from_studio.get_user_domain().as_dict(),
        data_local.get_user_domain().as_dict(),
    )
    studio_domain = Domain.from_dict(new_domain_data)

    if not studio_domain.is_empty():
        studio_domain.persist(studio_domain_path)
    else:
        structlogger.warning(
            "studio.download.merge_directory_domain",
            event_info="No additional domain data found in Studio assistant.",
        )


def _merge_file_domain(
    domain_path: Path,
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
) -> None:
    """Merges domain data without overwriting when the domain is a file.

    Args:
        domain_path: The path to the domain file.
        data_from_studio: The Studio data importer.
        data_local: The local data importer.
    """
    domain_merged = data_local.get_user_domain().merge(
        data_from_studio.get_user_domain()
    )
    domain_merged.persist(domain_path)


def _merge_data_no_overwrite(
    data_paths: List[Path],
    handler: StudioDataHandler,
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
) -> None:
    """Merges NLU and flow data without overwriting existing data.

    Args:
        data_paths: The paths to the data files or directories.
        handler: The StudioDataHandler instance.
        data_from_studio: The Studio data importer.
        data_local: The local data importer.
    """
    if not data_paths:
        structlogger.warning(
            "studio.download.merge_data_no_overwrite.no_path",
            event_info="No data paths provided. Skipping data merge.",
        )
        return

    if len(data_paths) == 1:
        data_path = data_paths[0]
        if data_path.is_file():
            _merge_file_data_no_overwrite(
                data_path, handler, data_from_studio, data_local
            )
        elif data_path.is_dir():
            _merge_dir_data_no_overwrite(
                data_path, handler, data_from_studio, data_local
            )
        else:
            structlogger.warning(
                "studio.download.merge_data_no_overwrite.invalid_path",
                event_info=(
                    f"Provided path '{data_path}' is neither a file nor a directory."
                ),
            )
    else:
        # TODO: Handle multiple data paths.
        raise NotImplementedError("Multiple data paths are not supported yet.")


def _merge_file_data_no_overwrite(
    data_path: Path,
    handler: StudioDataHandler,
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
) -> None:
    """Merges NLU and flows data into a single file without overwriting.

    Args:
        data_path: Path to the single data file.
        handler: The StudioDataHandler instance.
        data_from_studio: The Studio data importer.
        data_local: The local data importer.
    """
    if handler.has_nlu():
        merged_nlu = data_local.get_nlu_data().merge(data_from_studio.get_nlu_data())
        merged_nlu.persist_nlu(str(data_path))
    if handler.has_flows():
        merged_flows = data_local.get_user_flows().merge(
            data_from_studio.get_user_flows()
        )
        YamlFlowsWriter.dump(merged_flows.underlying_flows, data_path)


def _merge_dir_data_no_overwrite(
    dir_path: Path,
    handler: StudioDataHandler,
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
) -> None:
    """Merges NLU and flows data into a single directory without overwriting.

    Args:
        dir_path: Path to the data directory.
        handler: The StudioDataHandler instance.
        data_from_studio: The Studio data importer.
        data_local: The local data importer.
    """
    if handler.has_nlu():
        nlu_path = dir_path / Path(STUDIO_NLU_FILENAME)
        _persist_nlu_diff(data_local, data_from_studio, nlu_path)
    if handler.has_flows():
        flows_path = dir_path / Path(STUDIO_FLOWS_FILENAME)
        _persist_flows_diff(data_local, data_from_studio, flows_path)


def _handle_download_with_overwrite(
    handler: StudioDataHandler, domain_path: Path, data_paths: List[Path]
) -> None:
    """Handles downloading and merging data when the user opts for overwrite.

    Args:
        handler: The StudioDataHandler instance.
        domain_path: The path to the domain file or directory.
        data_paths: The paths to the data files or directories.
    """
    data_from_studio, data_local = import_data_from_studio(
        handler, domain_path, data_paths
    )
    mapper = RasaPrimitiveStorageMapper(
        domain_path=domain_path, training_data_paths=data_paths
    )
    merge_domain_with_overwrite(data_from_studio, data_local, domain_path)
    merge_flows_with_overwrite(
        data_paths, handler, data_from_studio, data_local, mapper
    )


def _persist_nlu_diff(
    data_local: TrainingDataImporter,
    data_from_studio: TrainingDataImporter,
    data_path: Path,
) -> None:
    """Creates a new NLU file from the diff of local and studio data.

    Args:
        data_local: The local training data.
        data_from_studio: The training data from Rasa Studio.
        data_path: The path to the NLU file.
    """
    new_nlu_data = data_handler.create_new_nlu_from_diff(
        read_yaml(data_from_studio.get_nlu_data().nlu_as_yaml()),
        read_yaml(data_local.get_nlu_data().nlu_as_yaml()),
    )
    if new_nlu_data["nlu"]:
        pretty_write_nlu_yaml(new_nlu_data, data_path)
    else:
        structlogger.warning(
            "studio.download.persist_nlu_diff",
            event_info="No additional nlu data found.",
        )


def _persist_flows_diff(
    data_local: TrainingDataImporter,
    data_from_studio: TrainingDataImporter,
    data_path: Path,
) -> None:
    """Creates a new flows file from the diff of local and studio data.

    Args:
        data_local: The local training data.
        data_from_studio: The training data from Rasa Studio.
        data_path: The path to the flows file.
    """
    new_flows_data = data_handler.create_new_flows_from_diff(
        data_from_studio.get_user_flows().underlying_flows,
        data_local.get_user_flows().underlying_flows,
    )
    if new_flows_data:
        YamlFlowsWriter.dump(new_flows_data, data_path)
    else:
        structlogger.warning(
            "studio.download.persist_flows_diff",
            event_info="No additional flows data found.",
        )


def pretty_write_nlu_yaml(data: Dict, file: Path) -> None:
    """Writes the NLU YAML in a pretty way.

    Args:
        data: The data to write.
        file: The file to write to.
    """
    dumper = yaml.YAML()
    for item in data["nlu"]:
        if item.get("examples"):
            item["examples"] = LiteralScalarString(item["examples"])
    with file.open("w", encoding="utf-8") as outfile:
        dumper.dump(data, outfile)
