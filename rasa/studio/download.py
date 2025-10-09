import argparse
import shutil
from pathlib import Path
from typing import Dict

import questionary
import structlog
from ruamel import yaml
from ruamel.yaml.scalarstring import LiteralScalarString

import rasa.cli.utils
import rasa.shared.utils.cli
from rasa.core.config.configuration import Configuration
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.shared.core.flows.yaml_flows_io import FlowsList
from rasa.shared.nlu.training_data.training_data import (
    DEFAULT_TRAINING_DATA_OUTPUT_PATH,
)
from rasa.shared.utils.yaml import read_yaml, write_yaml
from rasa.studio.config import StudioConfig
from rasa.studio.constants import DOMAIN_FILENAME
from rasa.studio.data_handler import StudioDataHandler
from rasa.studio.prompts import handle_prompts
from rasa.studio.pull.data import _dump_flows_as_separate_files
from rasa.studio.utils import validate_argument_paths

structlogger = structlog.get_logger()


def handle_download(args: argparse.Namespace) -> None:
    """Download an assistant from Studio and store it in `<assistant_name>/`.

    Args:
        args: The command line arguments.
    """
    validate_argument_paths(args)
    assistant_name = args.assistant_name
    target_root = _prepare_target_directory(assistant_name)

    handler = StudioDataHandler(
        studio_config=StudioConfig.read_config(), assistant_name=assistant_name
    )
    handler.request_all_data()

    _handle_config(handler, target_root)
    _handle_endpoints(handler, target_root)
    _handle_domain(handler, target_root)
    _handle_data(handler, target_root)

    if prompts := handler.get_prompts():
        handle_prompts(prompts, target_root)

    structlogger.info(
        "studio.download.success",
        event_info=f"Downloaded assistant '{assistant_name}' from Studio.",
        assistant_name=assistant_name,
    )
    rasa.shared.utils.cli.print_success(
        f"Downloaded assistant '{assistant_name}' from Studio."
    )


def _prepare_target_directory(assistant_name: str) -> Path:
    """Create (or overwrite) the directory where everything is stored.

    Args:
        assistant_name: The name of the assistant to download.

    Returns:
        The path to the target directory where the assistant will be stored.
    """
    target_root = Path(assistant_name)

    if target_root.exists():
        overwrite = questionary.confirm(
            f"Directory '{assistant_name}' already exists. Overwrite it?"
        ).ask()
        if not overwrite:
            rasa.shared.utils.cli.print_error_and_exit("Download cancelled.")

        shutil.rmtree(target_root)

    target_root.mkdir(parents=True, exist_ok=True)
    return target_root


def _handle_config(handler: StudioDataHandler, root: Path) -> None:
    """Download and persist the assistant’s config file.

    Args:
        handler: The data handler to retrieve the config from.
        root: The root directory where the config file will be stored.
    """
    config_data = handler.get_config()
    if not config_data:
        rasa.shared.utils.cli.print_error_and_exit("No config data found.")

    config_path = root / DEFAULT_CONFIG_PATH
    config_path.write_text(config_data, encoding="utf-8")


def _handle_endpoints(handler: StudioDataHandler, root: Path) -> None:
    """Download and persist the assistant’s endpoints file.

    Args:
        handler: The data handler to retrieve the endpoints from.
        root: The root directory where the endpoints file will be stored.
    """
    endpoints_data = handler.get_endpoints()
    if not endpoints_data:
        rasa.shared.utils.cli.print_error_and_exit("No endpoints data found.")

    endpoints_path = root / DEFAULT_ENDPOINTS_PATH
    endpoints_path.write_text(endpoints_data, encoding="utf-8")

    Configuration.initialise_endpoints(endpoints_path=endpoints_path)


def _handle_domain(handler: StudioDataHandler, root: Path) -> None:
    """Persist the assistant’s domain file.

    Args:
        handler: The data handler to retrieve the domain from.
        root: The root directory where the domain file will be stored.
    """
    domain_yaml = handler.domain
    data = read_yaml(domain_yaml)
    target = root / DOMAIN_FILENAME
    write_yaml(
        data=data,
        target=target,
        should_preserve_key_order=True,
    )


def _handle_data(handler: StudioDataHandler, root: Path) -> None:
    """Persist NLU data and flows.

    Args:
        handler: The data handler to retrieve the NLU data and flows from.
        root: The root directory where the NLU data and flows will be stored.
    """
    data_path = root / DEFAULT_DATA_PATH
    data_path.mkdir(parents=True, exist_ok=True)

    if handler.has_nlu():
        nlu_yaml = handler.nlu
        nlu_data = read_yaml(nlu_yaml)
        if nlu_data.get("nlu"):
            pretty_write_nlu_yaml(
                nlu_data, data_path / DEFAULT_TRAINING_DATA_OUTPUT_PATH
            )

    if handler.has_flows():
        flows_yaml = handler.flows
        data = read_yaml(flows_yaml)
        flows_data = data.get("flows", {})
        flows_list = FlowsList.from_json(flows_data)
        _dump_flows_as_separate_files(flows_list.underlying_flows, data_path)


def pretty_write_nlu_yaml(data: Dict, file: Path) -> None:
    """Writes the NLU YAML in a pretty way."""
    dumper = yaml.YAML()
    if nlu_data := data.get("nlu"):
        for item in nlu_data:
            if item.get("examples"):
                item["examples"] = LiteralScalarString(item["examples"])
    with file.open("w", encoding="utf-8") as outfile:
        dumper.dump(data, outfile)
