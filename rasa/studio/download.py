import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import questionary
import structlog

import rasa.cli.utils
import rasa.shared.utils.cli
from rasa.shared.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.shared.core.domain import Domain
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
from rasa.studio.data_handler import (
    StudioDataHandler,
    import_data_from_studio,
)
from rasa.utils.mapper import RasaPrimitiveStorageMapper

logger = logging.getLogger(__name__)
structlogger = structlog.getLogger(__name__)


def _handle_file_overwrite(
    file_path: Optional[str], default_path: str, file_type: str
) -> Tuple[Optional[Path], bool]:
    """Handles the logic for determining whether to
    overwrite an existing file or create a new one.
    Works for config and endpoints at this moment

    Args:
        file_path (Optional[str]): The path to the file
                    provided by the user. Can be None.
        default_path (str): The default path to use if `file_path`
                    is None or invalid. Must be a file path.
        file_type (str): The type of the file (e.g., "config",
            "endpoints") for logging and messaging purposes.

    Returns:
        tuple[Optional[Path], bool]: A tuple containing the
                            resolved file path and a boolean
                        indicating whether to write the file.
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
        f"{file_type.capitalize()} file '{file_or_default_path}' "
        f"already exists. Do you want to overwrite it?"
    ).ask():
        write_file = True
        path = Path(file_or_default_path)
    return path, write_file


def _prepare_data_and_domain_paths(args: argparse.Namespace) -> Tuple[Path, List[Path]]:
    """Handles the logic for preparing the domain and data paths
    based on the provided arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        tuple[Path, list[Path]]: A tuple containing the domain path
                        and a list of data paths.
    """
    # prepare domain
    domain_path = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )
    domain_or_default_path = args.domain or DEFAULT_DOMAIN_PATH

    if domain_path is None:
        # If the path is None, use the provided domain path
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

    # prepare data
    data_paths = []

    for f in args.data:
        data_path = rasa.cli.utils.get_validated_path(
            f, "data", DEFAULT_DATA_PATH, none_is_valid=True
        )

        if data_path is None:
            # If the path is None, use the default data path
            data_path = Path(f)
            data_path.mkdir(parents=True, exist_ok=True)
        else:
            data_path = Path(data_path)

        if data_path.is_file() or data_path.is_dir():
            # If it's a file, add it directly
            data_paths.append(data_path)
        else:
            # If it doesn't exist, create the directory
            data_path.mkdir(parents=True, exist_ok=True)
            data_paths.append(data_path)

    # Remove duplicates while preserving order
    data_paths = list(dict.fromkeys(data_paths))

    return domain_path, data_paths


def handle_download(args: argparse.Namespace) -> None:
    handler = StudioDataHandler(
        studio_config=StudioConfig.read_config(), assistant_name=args.assistant_name[0]
    )
    handler.request_all_data()

    domain_path, data_paths = _prepare_data_and_domain_paths(args)

    # handle config and endpoints
    config_path, write_config = _handle_file_overwrite(
        args.config, DEFAULT_CONFIG_PATH, "config"
    )
    endpoints_path, write_endpoints = _handle_file_overwrite(
        args.endpoints, DEFAULT_ENDPOINTS_PATH, "endpoints"
    )

    # generate log message if we write the config or endpoints
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
        _handle_download_no_overwrite(
            handler=handler,
            domain_path=domain_path,
            data_paths=data_paths,
        )
    else:
        _handle_download_with_overwrite(
            handler=handler,
            domain_path=domain_path,
            data_paths=data_paths,
        )


def _handle_download_no_overwrite(
    handler: StudioDataHandler,
    domain_path: Path,
    data_paths: List[Path],
) -> None:
    data_from_studio, data_original = import_data_from_studio(
        handler, domain_path, data_paths
    )

    if domain_path.is_dir():
        studio_domain_path = domain_path / STUDIO_DOMAIN_FILENAME
        new_domain_data = data_handler.combine_domains(
            data_from_studio.get_user_domain().as_dict(),
            data_original.get_user_domain().as_dict(),
        )
        studio_domain = Domain.from_dict(new_domain_data)
        if not studio_domain.is_empty():
            studio_domain.persist(studio_domain_path)
        else:
            logger.warning("No additional domain data found.")
    elif domain_path.is_file():
        domain_merged = data_original.get_user_domain().merge(
            data_from_studio.get_user_domain()
        )
        domain_merged.persist(domain_path)

    if len(data_paths) == 1 and data_paths[0].is_file():
        data_path = data_paths[0]
        if handler.has_nlu():
            data_nlu = data_original.get_nlu_data().merge(
                data_from_studio.get_nlu_data()
            )
            data_nlu.persist_nlu(data_path)
        if handler.has_flows():
            data_flows = data_original.get_user_flows().merge(
                data_from_studio.get_user_flows()
            )
            YamlFlowsWriter.dump(data_flows.underlying_flows, data_path)

    elif len(data_paths) == 1 and data_paths[0].is_dir():
        if handler.has_nlu():
            data_path = data_paths[0] / Path(STUDIO_NLU_FILENAME)
            _persist_nlu_diff(data_original, data_from_studio, data_path)
        if handler.has_flows():
            data_path = data_paths[0] / Path(STUDIO_FLOWS_FILENAME)
            _persist_flows_diff(data_original, data_from_studio, data_path)
    else:
        if data_paths[0].is_dir():
            logger.info(f"Saving data to {data_paths[0]}.")
            data_path = data_paths[0] / Path(STUDIO_NLU_FILENAME)
        else:
            logger.info(f"Saving data to {STUDIO_NLU_FILENAME}.")
            data_path = Path(STUDIO_NLU_FILENAME)
        if handler.has_nlu():
            _persist_nlu_diff(data_original, data_from_studio, data_path)
        if handler.has_flows():
            _persist_flows_diff(data_original, data_from_studio, data_path)


def _persist_nlu_diff(
    data_original: TrainingDataImporter,
    data_from_studio: TrainingDataImporter,
    data_path: Path,
) -> None:
    """Creates a new nlu file from the diff of original and studio data."""
    new_nlu_data = data_handler.create_new_nlu_from_diff(
        read_yaml(data_from_studio.get_nlu_data().nlu_as_yaml()),
        read_yaml(data_original.get_nlu_data().nlu_as_yaml()),
    )
    if new_nlu_data["nlu"]:
        pretty_write_nlu_yaml(new_nlu_data, data_path)
    else:
        logger.warning("No additional nlu data found.")


def _persist_flows_diff(
    data_original: TrainingDataImporter,
    data_from_studio: TrainingDataImporter,
    data_path: Path,
) -> None:
    """Creates a new flows file from the diff of original and studio data."""
    new_flows_data = data_handler.create_new_flows_from_diff(
        data_from_studio.get_user_flows().underlying_flows,
        data_original.get_user_flows().underlying_flows,
    )
    if new_flows_data:
        YamlFlowsWriter.dump(new_flows_data, data_path)
    else:
        logger.warning("No additional flows data found.")


def pretty_write_nlu_yaml(data: Dict, file: Path) -> None:
    """Writes the nlu yaml in a pretty way.

    Args:
        data: The data to write.
        file: The file to write to.
    """
    from ruamel import yaml
    from ruamel.yaml.scalarstring import LiteralScalarString

    dumper = yaml.YAML()
    for item in data["nlu"]:
        if item.get("examples"):
            item["examples"] = LiteralScalarString(item["examples"])

    with file.open("w", encoding="utf-8") as outfile:
        dumper.dump(data, outfile)


def _handle_download_with_overwrite(
    handler: StudioDataHandler,
    domain_path: Path,
    data_paths: List[Path],
) -> None:
    data_from_studio, data_original = import_data_from_studio(
        handler, domain_path, data_paths
    )
    mapper = RasaPrimitiveStorageMapper(
        domain_path=domain_path, training_data_paths=data_paths
    )

    if domain_path.is_file():
        domain_merged = data_from_studio.get_user_domain().merge(
            data_original.get_user_domain()
        )
        domain_merged.persist(domain_path)
    elif domain_path.is_dir():
        default = domain_path / Path(STUDIO_DOMAIN_FILENAME)
        studio_domain = data_from_studio.get_user_domain()

        paths = get_domain_path(domain_path, data_from_studio, mapper)

        _persist_domain_part(
            {"intents": studio_domain.as_dict().get("intents", [])},
            default,
            paths["intent_path"],
        )
        _persist_domain_part(
            {"entities": studio_domain.as_dict().get("entities", [])},
            default,
            paths["entities_path"],
        )
        _persist_domain_part(
            {"actions": studio_domain.as_dict().get("actions", [])},
            default,
            paths["action_path"],
        )
        _persist_domain_part(
            {"slots": studio_domain.as_dict().get("slots", {})},
            default,
            paths["slot_path"],
        )
        _persist_domain_part(
            {"responses": studio_domain.as_dict().get("responses", {})},
            default,
            paths["response_path"],
        )

    if len(data_paths) == 1 and data_paths[0].is_file():
        if handler.has_nlu():
            nlu_data_merged = data_from_studio.get_nlu_data().merge(
                data_original.get_nlu_data()
            )
            nlu_data_merged.persist_nlu(data_paths[0])
        if handler.has_flows():
            flows_data_merged = data_from_studio.get_user_flows().merge(
                data_original.get_user_flows()
            )
            YamlFlowsWriter.dump(flows_data_merged.underlying_flows, data_paths[0])
    elif len(data_paths) == 1 and data_paths[0].is_dir():
        data_path = data_paths[0]
        paths = get_training_path(data_path, data_original, mapper)
        if handler.has_nlu():
            nlu_data = data_from_studio.get_nlu_data()
            if paths["nlu_path"].exists():
                nlu_file = TrainingDataImporter.load_from_dict(
                    training_data_paths=[str(paths["nlu_path"])]
                )
                nlu_data = nlu_data.merge(nlu_file.get_nlu_data())
            pretty_write_nlu_yaml(read_yaml(nlu_data.nlu_as_yaml()), paths["nlu_path"])
        if handler.has_flows():
            flows_data = data_from_studio.get_user_flows()
            if paths["flow_path"].exists():
                flows_file = TrainingDataImporter.load_from_dict(
                    training_data_paths=[str(paths["flow_path"])]
                )
                flows_data = flows_data.merge(flows_file.get_user_flows())
            YamlFlowsWriter.dump(flows_data.underlying_flows, paths["flow_path"])
    else:
        #  TODO: we are not handling the case of multiple data paths?
        raise NotImplementedError("Multiple data paths are not supported yet.")


def _persist_domain_part(
    domain_part: Dict[str, Any], default: Path, path: Optional[Path]
) -> None:
    domain = Domain.from_dict(domain_part)
    if (path is not None) and path.exists():
        domain = Domain.from_file(path).merge(domain, override=True)
        domain.persist(path)
    else:
        domain.persist(default)


def get_training_path(
    path: Path,
    data_original: TrainingDataImporter,
    mapper: RasaPrimitiveStorageMapper,
) -> Dict[str, Path]:
    nlu_paths = set()
    flow_paths = set()
    for intent in data_original.get_nlu_data().intents:
        for path in mapper.get_file(intent, "intents").get("training", []):
            nlu_paths.add(path)
    flows = [flow.id for flow in data_original.get_user_flows().underlying_flows]
    for flow in flows:
        for path in mapper.get_file(flow, "flows").get("training", []):
            flow_paths.add(path)

    return {
        "nlu_path": _select_path(nlu_paths, "nlu", path, STUDIO_NLU_FILENAME),
        "flow_path": _select_path(flow_paths, "flows", path, STUDIO_FLOWS_FILENAME),
    }


def get_domain_path(
    domain_path: Path,
    data_from_studio: TrainingDataImporter,
    mapper: RasaPrimitiveStorageMapper,
) -> Dict[str, Path]:
    intent_paths = set()
    entities_paths = set()
    slot_paths = set()
    action_paths = set()
    response_paths = set()
    # nlu-based
    for intent in data_from_studio.get_domain().intents:
        for path in mapper.get_file(intent, "intents").get("domain", []):
            intent_paths.add(path)
    for entity in data_from_studio.get_domain().entities:
        for path in mapper.get_file(entity, "entities").get("domain", []):
            entities_paths.add(path)
    # flow-based
    slot_list = [slot.name for slot in data_from_studio.get_domain().slots]
    for slot in slot_list:
        for path in mapper.get_file(slot, "slots").get("domain", []):
            slot_paths.add(path)
    for action in data_from_studio.get_domain().action_names_or_texts:
        for path in mapper.get_file(action, "actions").get("domain", []):
            action_paths.add(path)
    for response in data_from_studio.get_domain().responses:
        for path in mapper.get_file(response, "responses").get("domain", []):
            response_paths.add(path)

    return {
        "intent_path": _select_path(
            intent_paths, "intents", domain_path, STUDIO_DOMAIN_FILENAME
        ),
        "entities_path": _select_path(
            entities_paths, "entities", domain_path, STUDIO_DOMAIN_FILENAME
        ),
        "slot_path": _select_path(
            slot_paths, "slots", domain_path, STUDIO_DOMAIN_FILENAME
        ),
        "action_path": _select_path(
            action_paths, "actions", domain_path, STUDIO_DOMAIN_FILENAME
        ),
        "response_path": _select_path(
            response_paths, "responses", domain_path, STUDIO_DOMAIN_FILENAME
        ),
    }


def _select_path(
    paths: Set[Path], primitive_type: str, default_path: Path, default: str
) -> Path:
    if len(paths) == 1:
        path = paths.pop()
    elif len(paths) > 1:
        path = paths.pop()
        logger.warning(
            f"Saving {primitive_type} to {path}."
            f"Please keep Studio related {primitive_type} in a single file."
        )
    else:  # no path in paths
        path = default_path / Path(default)
        logger.info(f"Saving {primitive_type} to {path}.")
    return path
