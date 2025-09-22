from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple

import structlog

import rasa.cli.utils
import rasa.shared.utils.cli
from rasa.cli.validation.config_path_validation import get_validated_path
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DOMAIN_PATHS,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.io import write_text_file
from rasa.studio.config import StudioConfig
from rasa.studio.data_handler import StudioDataHandler, import_data_from_studio
from rasa.studio.link import read_assistant_name
from rasa.studio.pull.data import merge_flows_in_directory, merge_nlu_in_directory
from rasa.studio.pull.domains import merge_domain
from rasa.studio.utils import validate_argument_paths
from rasa.utils.mapper import RasaPrimitiveStorageMapper

structlogger = structlog.get_logger(__name__)


def handle_pull(args: argparse.Namespace) -> None:
    """Pull the complete assistant and overwrite all local files.

    Args:
        args: The command line arguments.
    """
    validate_argument_paths(args)
    handler = _create_studio_handler()
    handler.request_all_data()

    _pull_config_file(handler, args.config or DEFAULT_CONFIG_PATH)
    _pull_endpoints_file(handler, args.endpoints or DEFAULT_ENDPOINTS_PATH)

    domain_path, data_path = _prepare_data_and_domain_paths(args)
    _merge_domain_and_data(handler, domain_path, data_path)

    structlogger.info(
        "studio.push.success",
        event_info="Pulled assistant data from Studio.",
    )
    rasa.shared.utils.cli.print_success("Pulled assistant data from Studio.")


def handle_pull_config(args: argparse.Namespace) -> None:
    """Pull nothing but the assistant's `config.yml`.

    Args:
        args: The command line arguments.
    """
    validate_argument_paths(args)
    handler = _create_studio_handler()
    handler.request_all_data()

    _pull_config_file(handler, args.config or DEFAULT_CONFIG_PATH)

    structlogger.info(
        "studio.push.success",
        event_info="Pulled assistant data from Studio.",
    )
    rasa.shared.utils.cli.print_success("Pulled assistant data from Studio.")


def handle_pull_endpoints(args: argparse.Namespace) -> None:
    """Pull nothing but the assistant's `endpoints.yml`.

    Args:
        args: The command line arguments.
    """
    validate_argument_paths(args)
    handler = _create_studio_handler()
    handler.request_all_data()

    _pull_endpoints_file(handler, args.endpoints or DEFAULT_ENDPOINTS_PATH)
    structlogger.info(
        "studio.push.success",
        event_info="Pulled assistant data from Studio.",
    )
    rasa.shared.utils.cli.print_success("Pulled assistant data from Studio.")


def _create_studio_handler() -> StudioDataHandler:
    """Return an initialised StudioDataHandler for the linked assistant.

    Returns:
        An instance of `StudioDataHandler` for the linked assistant.
    """
    assistant_name = read_assistant_name()
    return StudioDataHandler(
        studio_config=StudioConfig.read_config(), assistant_name=assistant_name
    )


def _pull_config_file(handler: StudioDataHandler, target_path: str | Path) -> None:
    """Pull the assistant's `config.yml` file and write it to the specified path.

    Args:
        handler: The data handler to retrieve the config from.
        target_path: The path where the config file should be written.
    """
    config_yaml = handler.get_config()
    if not config_yaml:
        rasa.shared.utils.cli.print_error_and_exit("No config data found in assistant.")

    _write_text(config_yaml, target_path)


def _pull_endpoints_file(handler: StudioDataHandler, target_path: str | Path) -> None:
    """Pull the assistant's `endpoints.yml` file and write it to the specified path.

    Args:
        handler: The data handler to retrieve the endpoints from.
        target_path: The path where the endpoints file should be written.
    """
    endpoints_yaml = handler.get_endpoints()
    if not endpoints_yaml:
        rasa.shared.utils.cli.print_error_and_exit(
            "No endpoints data found in assistant."
        )

    _write_text(endpoints_yaml, target_path)


def _prepare_data_and_domain_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Prepars the domain and data paths based on the provided arguments.

    Args:
        args: The command line arguments.

    Returns:
        A tuple containing the domain path and a data path.
    """
    # Prepare domain path.
    domain_path = get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATHS, none_is_valid=True
    )
    domain_or_default_path = args.domain or DEFAULT_DOMAIN_PATH

    if domain_path is None:
        domain_path = Path(domain_or_default_path)
        domain_path.touch()

    if isinstance(domain_path, str):
        domain_path = Path(domain_path)

    data_path = get_validated_path(
        args.data, "data", DEFAULT_DATA_PATH, none_is_valid=True
    )

    data_path = Path(data_path or args.data)
    if not (data_path.is_file() or data_path.is_dir()):
        data_path.mkdir(parents=True, exist_ok=True)

    return domain_path, data_path


def _merge_domain_and_data(
    handler: StudioDataHandler, domain_path: Path, data_path: Path
) -> None:
    """Merge local assistant data with Studio assistant data.

    Args:
        handler: The data handler to retrieve the assistant data from.
        domain_path: The path to the local domain file or directory.
        data_path: The path to the local training data file or directory.
    """
    data_from_studio, data_local = import_data_from_studio(
        handler, domain_path, data_path
    )
    mapper = RasaPrimitiveStorageMapper(
        domain_path=domain_path, training_data_paths=[data_path]
    )

    merge_domain(data_from_studio, data_local, domain_path)
    merge_data(data_path, handler, data_from_studio, data_local, mapper)


def merge_data(
    data_path: Path,
    handler: Any,
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
    mapper: RasaPrimitiveStorageMapper,
) -> None:
    """
    Merges flows data from a file or directory.

    Args:
        data_path: List of paths to the training data.
        handler: The StudioDataHandler instance.
        data_from_studio: The TrainingDataImporter instance for Studio data.
        data_local: The TrainingDataImporter instance for local data.
        mapper: The RasaPrimitiveStorageMapper instance for mapping.
    """
    if not data_path.is_file() and not data_path.is_dir():
        raise ValueError("Provided data path is neither a file nor a directory.")

    if handler.has_nlu():
        if data_path.is_file():
            nlu_data_merged = data_from_studio.get_nlu_data().merge(
                data_local.get_nlu_data()
            )
            nlu_data_merged.persist_nlu(data_path)
        else:
            merge_nlu_in_directory(
                data_from_studio,
                data_local,
                data_path,
                mapper,
            )

    if handler.has_flows():
        flows_root = data_path.parent if data_path.is_file() else data_path
        merge_flows_in_directory(
            data_from_studio,
            flows_root,
            mapper,
        )


def _write_text(content: str, target: str | Path) -> None:
    """Write `content` to `target`, ensuring parent directories exist.

    Args:
        content: The content to write to the file.
        target: The path where the content should be written.
    """
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_file(content, path, encoding="utf-8")
