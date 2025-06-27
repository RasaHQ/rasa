from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Text

import structlog

import rasa.shared.utils.cli
from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.utils.io import read_file
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string
from rasa.studio.config import StudioConfig
from rasa.studio.link import get_studio_config, read_assistant_name
from rasa.studio.upload import (
    build_import_request,
    make_request,
    run_validation,
)
from rasa.studio.utils import validate_argument_paths

structlogger = structlog.get_logger(__name__)


def _send_to_studio(
    assistant_name: Text,
    payload_parts: Dict[Text, Text],
    studio_cfg: StudioConfig,
) -> None:
    """Build the GraphQL request and send it.

    Args:
        assistant_name: The name of the assistant.
        payload_parts: The parts of the payload to send.
        studio_cfg: The StudioConfig object.
    """
    graphql_req = build_import_request(
        assistant_name=assistant_name,
        flows_yaml=payload_parts.get("flows"),
        domain_yaml=payload_parts.get("domain"),
        config_yaml=payload_parts.get("config"),
        endpoints=payload_parts.get("endpoints"),
        nlu_yaml=payload_parts.get("nlu"),
    )
    verify = not studio_cfg.disable_verify
    result = make_request(studio_cfg.studio_url, graphql_req, verify)
    if not result.was_successful:
        rasa.shared.utils.cli.print_error_and_exit(result.message)

    structlogger.info(
        "studio.push.success",
        event_info=f"Pushed data to assistant '{assistant_name}'.",
        assistant_name=assistant_name,
    )
    rasa.shared.utils.cli.print_success(f"Pushed data to assistant '{assistant_name}'.")


def handle_push(args: argparse.Namespace) -> None:
    """Push the entire assistant.

    Args:
        args: The command line arguments.
    """
    validate_argument_paths(args)
    studio_cfg = get_studio_config()

    run_validation(args)

    importer = TrainingDataImporter.load_from_dict(
        domain_path=args.domain,
        config_path=args.config,
        expand_env_vars=False,
    )

    domain_yaml = dump_obj_as_yaml_to_string(importer.get_user_domain().as_dict())
    config_yaml = read_file(Path(args.config))
    endpoints_yaml = read_file(Path(args.endpoints))

    flow_importer = TrainingDataImporter.load_from_dict(
        training_data_paths=args.data, expand_env_vars=False
    )
    flows_yaml = YamlFlowsWriter().dumps(flow_importer.get_user_flows())

    nlu_importer = TrainingDataImporter.load_from_dict(
        training_data_paths=args.data, expand_env_vars=False
    )
    nlu_yaml = RasaYAMLWriter().dumps(nlu_importer.get_nlu_data())

    assistant_name = read_assistant_name()
    _send_to_studio(
        assistant_name,
        {
            "flows": flows_yaml,
            "domain": domain_yaml,
            "config": config_yaml,
            "endpoints": endpoints_yaml,
            "nlu": nlu_yaml,
        },
        studio_cfg,
    )


def handle_push_config(args: argparse.Namespace) -> None:
    """Push only the assistant configuration (config.yml).

    Args:
        args: The command line arguments.
    """
    studio_cfg = get_studio_config()
    assistant_name = read_assistant_name()

    config_yaml = read_file(Path(args.config))
    if not config_yaml:
        rasa.shared.utils.cli.print_error_and_exit(
            "No configuration data was found in the assistant."
        )

    _send_to_studio(assistant_name, {"config": config_yaml}, studio_cfg)


def handle_push_endpoints(args: argparse.Namespace) -> None:
    """Push only the endpoints configuration (endpoints.yml).

    Args:
        args: The command line arguments.
    """
    studio_cfg = get_studio_config()
    assistant_name = read_assistant_name()

    endpoints_yaml = read_file(Path(args.endpoints))
    if not endpoints_yaml:
        rasa.shared.utils.cli.print_error_and_exit(
            "No endpoints data was found in the assistant."
        )

    _send_to_studio(assistant_name, {"endpoints": endpoints_yaml}, studio_cfg)
