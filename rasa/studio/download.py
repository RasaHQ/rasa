import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import questionary
import structlog

import rasa.cli.utils
import rasa.shared.utils.cli
from rasa.shared.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATHS,
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


def handle_download(args: argparse.Namespace) -> None:
    handler = StudioDataHandler(
        studio_config=StudioConfig.read_config(), assistant_name=args.assistant_name[0]
    )
    handler.request_all_data()
    domain_path = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATHS, none_is_valid=True
    )
    domain_path = Path(domain_path)

    config_file_exists_in_path = rasa.cli.utils.get_validated_path(
        args.config, "config", DEFAULT_CONFIG_PATH, none_is_valid=True
    )

    if config_file_exists_in_path:
        config_path = Path(config_file_exists_in_path)
    else:
        config_path = Path(args.config)

        if config_path.is_dir():
            config_path = config_path / DEFAULT_CONFIG_PATH
        elif not config_path.parent.exists():
            raise ValueError(
                f"Directory '{config_path.parent}' for " f"config file does not exist."
            )

    endpoints_file_exists_in_path = rasa.cli.utils.get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, none_is_valid=True
    )

    if endpoints_file_exists_in_path:
        endpoints_path = Path(endpoints_file_exists_in_path)
    else:
        endpoints_path = Path(args.endpoints)
        if endpoints_path.is_dir():
            endpoints_path = endpoints_path / DEFAULT_ENDPOINTS_PATH
        elif not endpoints_path.parent.exists():
            raise ValueError(
                f"Directory '{endpoints_path.parent}' for "
                f"endpoints file does not exist."
            )

    data_paths = [
        Path(
            rasa.cli.utils.get_validated_path(
                f, "data", DEFAULT_DATA_PATH, none_is_valid=False
            )
        )
        for f in args.data
    ]

    _handle_download_config_and_endpoints(
        handler=handler,
        config_path=config_path,
        endpoints_path=endpoints_path,
    )

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


def _handle_download_config_and_endpoints(
    handler: StudioDataHandler,
    config_path: Path,
    endpoints_path: Path,
) -> None:
    """This method handles the download of the config and endpoints files from Studio.
    If the files already exist in the specified paths, the user is prompted to confirm
    the overwriting of these files. If a user chooses not to overwrite, download skips
    these files but the other data is still overwritten --overwrite flag doesn't apply

    Args:
        handler: StudioDataHandler object
        config_path: Path to the existing or new config file
        endpoints_path: Path to the existing or new endpoints file
    """
    write_config = False
    write_endpoints = False
    if config_path.is_file():
        if questionary.confirm(
            f"Config file '{config_path}' already exists. "
            f"Do you want to overwrite it?"
        ).ask():
            write_config = True

    if endpoints_path.is_file():
        if questionary.confirm(
            f"Endpoints file '{endpoints_path}' already exists. "
            f"Do you want to overwrite it?"
        ).ask():
            write_endpoints = True

    config_data = handler.get_config()
    if not config_data:
        raise ValueError("No config data found.")

    endpoints_data = handler.get_endpoints()
    if not endpoints_data:
        raise ValueError("No endpoints data found.")

    structlogger.info(
        "studio.download.config_endpoints",
        event_info="Downloaded config and endpoints files from Rasa Studio.",
    )

    if write_config:
        with open(config_path, "w") as f:
            f.write(config_data)
    if write_endpoints:
        with open(endpoints_path, "w") as f:
            f.write(endpoints_data)


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
            data_from_studio.get_domain().as_dict(),
            data_original.get_domain().as_dict(),
        )
        studio_domain = Domain.from_dict(new_domain_data)
        if not studio_domain.is_empty():
            studio_domain.persist(studio_domain_path)
        else:
            logger.warning("No additional domain data found.")
    elif domain_path.is_file():
        domain_merged = data_original.get_domain().merge(data_from_studio.get_domain())
        domain_merged.persist(domain_path)

    if len(data_paths) == 1 and data_paths[0].is_file():
        data_path = data_paths[0]
        if handler.has_nlu():
            data_nlu = data_original.get_nlu_data().merge(
                data_from_studio.get_nlu_data()
            )
            data_nlu.persist_nlu(data_path)
        if handler.has_flows():
            data_flows = data_original.get_flows().merge(data_from_studio.get_flows())
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
        data_from_studio.get_flows().underlying_flows,
        data_original.get_flows().underlying_flows,
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
        domain_merged = data_from_studio.get_domain().merge(data_original.get_domain())
        domain_merged.persist(domain_path)
    elif domain_path.is_dir():
        default = domain_path / Path(STUDIO_DOMAIN_FILENAME)
        studio_domain = data_from_studio.get_domain()

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
            flows_data_merged = data_from_studio.get_flows().merge(
                data_original.get_flows()
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
            flows_data = data_from_studio.get_flows()
            if paths["flow_path"].exists():
                flows_file = TrainingDataImporter.load_from_dict(
                    training_data_paths=[str(paths["flow_path"])]
                )
                flows_data = flows_data.merge(flows_file.get_flows())
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
    flows = [flow.id for flow in data_original.get_flows().underlying_flows]
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
