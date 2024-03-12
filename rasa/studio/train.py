import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import rasa.cli.utils
import rasa.shared.utils.cli
from rasa.cli.train import (
    _model_for_finetuning,
    extract_core_additional_arguments,
    extract_nlu_additional_arguments,
)
from rasa.shared.constants import (
    CONFIG_MANDATORY_KEYS,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
)
from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import read_yaml, write_yaml
from rasa.utils.common import get_temp_dir_name

from rasa.studio.config import StudioConfig
from rasa.studio.data_handler import (
    DataDiffGenerator,
    StudioDataHandler,
    import_data_from_studio,
)

logger = logging.getLogger(__name__)


def handle_train(args: argparse.Namespace) -> Optional[str]:
    from rasa import train as train_all

    handler = StudioDataHandler(
        studio_config=StudioConfig.read_config(), assistant_name=args.assistant_name[0]
    )
    if args.entities or args.intents:
        handler.request_data(args.intents, args.entities)
    else:
        handler.request_all_data()

    domain = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )
    config = rasa.cli.utils.get_validated_config(args.config, CONFIG_MANDATORY_KEYS)
    data_form_studio, data_original = import_data_from_studio(
        handler, domain, args.data
    )

    domain = data_original.get_domain().merge(data_form_studio.get_domain())  # type: ignore[assignment]  # noqa: E501

    domain_file = _create_temp_file(read_yaml(domain.as_yaml()), "domain.yml")  # type: ignore[union-attr]  # noqa: E501

    trining_file = make_training_file(handler, data_form_studio, data_original)

    training_files = [
        rasa.cli.utils.get_validated_path(
            f, "data", DEFAULT_DATA_PATH, none_is_valid=True
        )
        for f in args.data
    ]
    training_files.append(trining_file)

    training_result = train_all(
        domain=str(domain_file),
        config=config,
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


def make_training_file(
    handler: StudioDataHandler,
    data_form_studio: TrainingDataImporter,
    data_original: TrainingDataImporter,
) -> Path:
    """Create training file from studio data and original data.

    Args:
        handler (StudioDataHandler): data handler with studio config
        data_form_studio (TrainingDataImporter): studio data
        data_original (TrainingDataImporter): original data

    Returns:
        Path: path to training file
    """
    if handler.nlu_assistant:
        # nlu has deduplication
        train_data = (
            data_original.get_nlu_data()
            .merge(data_form_studio.get_nlu_data())
            .nlu_as_yaml()
        )
        training_file = _create_temp_file(read_yaml(train_data), "nlu.yml")
    else:
        diff = DataDiffGenerator(
            original_flows=data_original.get_flows().underlying_flows,
            studio_flows=data_form_studio.get_flows().underlying_flows,
        )
        diff_flows = diff.create_new_flows_from_diff()
        tmp_dir = get_temp_dir_name()
        training_file = Path(tmp_dir, "flows.yml")
        YamlFlowsWriter.dump(diff_flows, training_file)
    return training_file


def _create_temp_file(data: Any, name: str) -> Path:
    tmp_dir = get_temp_dir_name()
    file = Path(tmp_dir, name)
    write_yaml(data, file)
    return file
