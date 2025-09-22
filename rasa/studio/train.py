import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from rasa.cli.train import (
    _model_for_finetuning,
    extract_core_additional_arguments,
    extract_nlu_additional_arguments,
)
from rasa.cli.validation.config_path_validation import (
    get_validated_config,
    get_validated_path,
)
from rasa.shared.constants import (
    CONFIG_MANDATORY_KEYS,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
)
from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import read_yaml, write_yaml
from rasa.studio import data_handler
from rasa.studio.config import StudioConfig
from rasa.studio.data_handler import (
    StudioDataHandler,
    import_data_from_studio,
)
from rasa.utils.common import get_temp_dir_name

logger = logging.getLogger(__name__)


def handle_train(args: argparse.Namespace) -> Optional[str]:
    from rasa.api import train as train_all

    handler = StudioDataHandler(
        studio_config=StudioConfig.read_config(), assistant_name=args.assistant_name
    )
    if args.entities or args.intents:
        handler.request_data(args.intents, args.entities)
    else:
        handler.request_all_data()

    domain = get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )
    config = get_validated_config(args.config, CONFIG_MANDATORY_KEYS)
    data_form_studio, data_original = import_data_from_studio(
        handler, domain, args.data
    )

    domain = data_original.get_domain().merge(data_form_studio.get_domain())  # type: ignore[assignment]

    domain_file = _create_temp_file(read_yaml(domain.as_yaml()), "domain.yml")  # type: ignore[union-attr]

    studio_training_files = make_training_files(
        handler, data_form_studio, data_original
    )

    training_files = [
        get_validated_path(f, "data", DEFAULT_DATA_PATH, none_is_valid=True)
        for f in args.data
    ]
    training_files.extend(studio_training_files)

    training_result = train_all(
        domain=str(domain_file),
        config=config,
        endpoints=args.endpoints,
        training_files=args.data,
        output=args.out,
        dry_run=args.dry_run,
        force_training=args.force,
        fixed_model_name=args.fixed_model_name,
        persist_nlu_training_data=args.persist_nlu_data,
        core_additional_arguments={
            **extract_core_additional_arguments(args),
        },
        nlu_additional_arguments=extract_nlu_additional_arguments(args),
        model_to_finetune=_model_for_finetuning(args),
        finetuning_epoch_fraction=args.epoch_fraction,
    )

    if training_result.code != 0:
        sys.exit(training_result.code)

    return training_result.model


def make_training_files(
    handler: StudioDataHandler,
    data_form_studio: TrainingDataImporter,
    data_original: TrainingDataImporter,
) -> List[Path]:
    """Create training file from studio data and original data.

    Args:
        handler (StudioDataHandler): data handler with studio config
        data_form_studio (TrainingDataImporter): studio data
        data_original (TrainingDataImporter): original data

    Returns:
        List[Path]: list of training files
    """
    training_files = []
    if handler.has_nlu():
        # nlu has deduplication
        train_data = (
            data_original.get_nlu_data()
            .merge(data_form_studio.get_nlu_data())
            .nlu_as_yaml()
        )
        training_file = _create_temp_file(read_yaml(train_data), "nlu.yml")
        training_files.append(training_file)

    if handler.has_flows():
        diff_flows = data_handler.create_new_flows_from_diff(
            data_form_studio.get_flows().underlying_flows,
            data_original.get_flows().underlying_flows,
        )
        tmp_dir = get_temp_dir_name()
        training_file = Path(tmp_dir, "flows.yml")
        YamlFlowsWriter.dump(diff_flows, training_file)
        training_files.append(training_file)

    return training_files


def _create_temp_file(data: Any, name: str) -> Path:
    tmp_dir = get_temp_dir_name()
    file = Path(tmp_dir, name)
    write_yaml(data, file)
    return file
