import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Text, Union

import randomname
import structlog

import rasa.engine.validation
import rasa.model
import rasa.shared.constants
import rasa.shared.exceptions
import rasa.shared.utils.cli
import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.utils.common
from rasa import telemetry
from rasa.engine.caching import LocalTrainingCache
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.components import FingerprintStatus
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.nlu.persistor import StorageType
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.data import TrainingType
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter

CODE_NEEDS_TO_BE_RETRAINED = 0b0001
CODE_FORCED_TRAINING = 0b1000
CODE_NO_NEED_TO_TRAIN = 0b0000

structlogger = structlog.get_logger()


class TrainingResult(NamedTuple):
    """Holds information about the results of training."""

    model: Optional[Text] = None
    code: int = 0
    dry_run_results: Optional[Dict[Text, Union[FingerprintStatus, Any]]] = None


def _dry_run_result(
    fingerprint_results: Dict[Text, Union[FingerprintStatus, Any]],
    force_full_training: bool,
) -> TrainingResult:
    """Returns a dry run result.

    Args:
        fingerprint_results: A result of fingerprint run..
        force_full_training: Whether the user used the `--force` flag to enforce a
            full retraining of the model.

    Returns:
        Result containing the return code and the fingerprint results.
    """
    if force_full_training:
        structlogger.warn(
            "model_training.force_full_training", event_info="The training was forced."
        )
        return TrainingResult(
            code=CODE_FORCED_TRAINING, dry_run_results=fingerprint_results
        )

    training_required = any(
        isinstance(result, FingerprintStatus) and not result.is_hit
        for result in fingerprint_results.values()
    )

    if training_required:
        structlogger.warn(
            "model_training.training_required",
            event_info="The model needs to be retrained.",
        )
        return TrainingResult(
            code=CODE_NEEDS_TO_BE_RETRAINED, dry_run_results=fingerprint_results
        )

    structlogger.info(
        "model_training.no_training_required",
        event_info=(
            "No training of components required "
            "(the responses might still need updating!)."
        ),
    )
    return TrainingResult(
        code=CODE_NO_NEED_TO_TRAIN, dry_run_results=fingerprint_results
    )


def get_unresolved_slots(domain: Domain, stories: StoryGraph) -> List[Text]:
    """Returns a list of unresolved slots.

    Args:
        domain: The domain.
        stories: The story graph.

    Returns:
        A list of unresolved slots.
    """
    return list(
        set(
            evnt.key
            for step in stories.story_steps
            for evnt in step.events
            if isinstance(evnt, SlotSet)
        )
        - set(slot.name for slot in domain.slots)
    )


def _check_unresolved_slots(domain: Domain, stories: StoryGraph) -> None:
    """Checks if there are any unresolved slots.

    Args:
        domain: The domain.
        stories: The story graph.

    Raises:
        `Sys exit` if there are any unresolved slots.

    Returns:
        `None` if there are no unresolved slots.
    """
    unresolved_slots = get_unresolved_slots(domain, stories)
    if unresolved_slots:
        structlogger.error(
            "model.training.check_unresolved_slots.not_in_domain",
            slots=unresolved_slots,
            event_info=(
                f"Unresolved slots found in stories/rules🚨 \n"
                f'Tried to set slots "{unresolved_slots}" that are not present in'
                f"your domain.\n Check whether they need to be added to the domain or "
                f"whether there is a spelling error."
            ),
        )
        sys.exit(1)


async def train(
    domain: Text,
    config: Text,
    training_files: Optional[Union[Text, List[Text]]],
    output: Text = rasa.shared.constants.DEFAULT_MODELS_PATH,
    dry_run: bool = False,
    force_training: bool = False,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    core_additional_arguments: Optional[Dict] = None,
    nlu_additional_arguments: Optional[Dict] = None,
    model_to_finetune: Optional[Text] = None,
    finetuning_epoch_fraction: float = 1.0,
    remote_storage: Optional[StorageType] = None,
) -> TrainingResult:
    """Trains a Rasa model (Core and NLU).

    Args:
        domain: Path to the domain file.
        config: Path to the config file.
        training_files: List of paths to training data files.
        output: Output directory for the trained model.
        dry_run: If `True` then no training will be done, and the information about
            whether the training needs to be done will be printed.
        force_training: If `True` retrain model even if data has not changed.
        fixed_model_name: Name of model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
            with the model.
        core_additional_arguments: Additional training parameters for core training.
        nlu_additional_arguments: Additional training parameters forwarded to training
            method of each NLU component.
        model_to_finetune: Optional path to a model which should be finetuned or
            a directory in case the latest trained model should be used.
        finetuning_epoch_fraction: The fraction currently specified training epochs
            in the model configuration which should be used for finetuning.
        remote_storage: The remote storage which should be used to store the model.

    Returns:
        An instance of `TrainingResult`.
    """
    file_importer = TrainingDataImporter.load_from_config(
        config, domain, training_files, core_additional_arguments
    )

    stories = file_importer.get_stories()
    flows = file_importer.get_flows()
    nlu_data = file_importer.get_nlu_data()

    training_type = TrainingType.BOTH

    if nlu_data.has_e2e_examples():
        rasa.shared.utils.common.mark_as_experimental_feature("end-to-end training")
        training_type = TrainingType.END_TO_END

    if stories.is_empty() and nlu_data.contains_no_pure_nlu_data() and flows.is_empty():
        structlogger.error(
            "model_training.train.no_training_data_found",
            event_info=(
                "No training data given. Please provide "
                "stories, flows or NLU data in "
                "order to train a Rasa model using the "
                "'--data' argument."
            ),
        )
        return TrainingResult(code=1)

    domain_object = file_importer.get_domain()
    if domain_object.is_empty():
        structlogger.warn(
            "model_training.train.domain_not_found",
            event_info=(
                "Core training was skipped because no "
                "valid domain file was found. Only an "
                "NLU-model was created. Please specify "
                "a valid domain using the '--domain' "
                "argument or check if the provided domain "
                "file exists."
            ),
        )
        training_type = TrainingType.NLU

    elif stories.is_empty() and flows.is_empty():
        structlogger.warn(
            "model_training.train.flows_and_stories_not_found",
            event_info=(
                "No stories or flows present. Just a " "Rasa NLU model will be trained."
            ),
        )
        training_type = TrainingType.NLU

    # We will train nlu if there are any nlu example, including from e2e stories.
    elif (
        nlu_data.contains_no_pure_nlu_data()
        and not nlu_data.has_e2e_examples()
        and flows.is_empty()
    ):
        structlogger.warn(
            "model_training.train.nlu_data_not_found",
            event_info="No NLU data present. No NLU model will be trained.",
        )
        training_type = TrainingType.CORE

    _check_unresolved_slots(domain_object, stories)

    with telemetry.track_model_training(file_importer, model_type="rasa"):
        return await _train_graph(
            file_importer,
            training_type=training_type,
            output_path=output,
            fixed_model_name=fixed_model_name,
            model_to_finetune=model_to_finetune,
            force_full_training=force_training,
            persist_nlu_training_data=persist_nlu_training_data,
            finetuning_epoch_fraction=finetuning_epoch_fraction,
            dry_run=dry_run,
            remote_storage=remote_storage,
            **(core_additional_arguments or {}),
            **(nlu_additional_arguments or {}),
        )


async def _train_graph(
    file_importer: TrainingDataImporter,
    training_type: TrainingType,
    output_path: Text,
    fixed_model_name: Text,
    model_to_finetune: Optional[Union[Text, Path]] = None,
    force_full_training: bool = False,
    dry_run: bool = False,
    remote_storage: Optional[StorageType] = None,
    **kwargs: Any,
) -> TrainingResult:
    if model_to_finetune:
        model_to_finetune = rasa.model.get_model_for_finetuning(model_to_finetune)
        if not model_to_finetune:
            structlogger.error(
                "model_training.train.finetuning_model_not_found",
                event_info=(
                    f"No model for finetuning found. Please make sure to either "
                    f"specify a path to a previous model or to have a finetunable "
                    f"model within the directory '{output_path}'."
                ),
            )
            sys.exit(1)

        rasa.shared.utils.common.mark_as_experimental_feature(
            "Incremental Training feature"
        )

    is_finetuning = model_to_finetune is not None

    config = file_importer.get_config()
    recipe = Recipe.recipe_for_name(config.get("recipe"))
    config, _missing_keys, _configured_keys = recipe.auto_configure(
        file_importer.get_config_file_for_auto_config(),
        config,
        training_type,
    )
    flows = file_importer.get_flows()
    domain = file_importer.get_domain()
    model_configuration = recipe.graph_config_for_recipe(
        config,
        kwargs,
        training_type=training_type,
        is_finetuning=is_finetuning,
    )
    rasa.engine.validation.validate(model_configuration)
    rasa.engine.validation.validate_coexistance_routing_setup(
        domain, model_configuration, flows
    )
    rasa.engine.validation.validate_flow_component_dependencies(
        flows, model_configuration
    )
    rasa.engine.validation.validate_command_generator_setup(model_configuration)

    tempdir_name = rasa.utils.common.get_temp_dir_name()
    # Use `TempDirectoryPath` instead of `tempfile.TemporaryDirectory` as this
    # leads to errors on Windows when the context manager tries to delete an
    # already deleted temporary directory (e.g. https://bugs.python.org/issue29982)
    with rasa.utils.common.TempDirectoryPath(tempdir_name) as temp_model_dir:
        model_storage = _create_model_storage(
            is_finetuning, model_to_finetune, Path(temp_model_dir)
        )
        cache = LocalTrainingCache()
        trainer = GraphTrainer(model_storage, cache, DaskGraphRunner)

        if dry_run:
            fingerprint_status = await trainer.fingerprint(
                model_configuration.train_schema, file_importer
            )
            return _dry_run_result(fingerprint_status, force_full_training)

        model_name = _determine_model_name(fixed_model_name, training_type)
        full_model_path = Path(output_path, model_name)

        with telemetry.track_model_training(
            file_importer, model_type=training_type.model_type
        ):
            await trainer.train(
                model_configuration,
                file_importer,
                full_model_path,
                force_retraining=force_full_training,
                is_finetuning=is_finetuning,
            )
            if remote_storage:
                push_model_to_remote_storage(full_model_path, remote_storage)
                full_model_path.unlink()
                structlogger.info(
                    "model_training.train.finished_training",
                    event_info=(
                        f"Your Rasa model {model_name} is trained "
                        f"and saved at remote storage provider '{remote_storage}'."
                    ),
                )
            else:
                structlogger.info(
                    "model_training.train.finished_training",
                    event_info=(
                        f"Your Rasa model is trained and saved at '{full_model_path}'."
                    ),
                )

        return TrainingResult(str(full_model_path), 0)


def _create_model_storage(
    is_finetuning: bool, model_to_finetune: Optional[Path], temp_model_dir: Path
) -> ModelStorage:
    if is_finetuning:
        model_storage, _ = LocalModelStorage.from_model_archive(
            temp_model_dir, model_to_finetune
        )
    else:
        model_storage = LocalModelStorage(temp_model_dir)

    return model_storage


def _determine_model_name(
    fixed_model_name: Optional[Text], training_type: TrainingType
) -> Text:
    if fixed_model_name:
        if not fixed_model_name.endswith(".tar.gz"):
            return f"{fixed_model_name}.tar.gz"
        return fixed_model_name

    prefix = ""
    if training_type in [TrainingType.CORE, TrainingType.NLU]:
        prefix = f"{training_type.model_type}-"

    time_format = "%Y%m%d-%H%M%S"
    return f"{prefix}{time.strftime(time_format)}-{randomname.get_name()}.tar.gz"


async def train_core(
    domain: Union[Domain, Text],
    config: Text,
    stories: Text,
    output: Text,
    fixed_model_name: Optional[Text] = None,
    additional_arguments: Optional[Dict] = None,
    model_to_finetune: Optional[Text] = None,
    finetuning_epoch_fraction: float = 1.0,
) -> Optional[Text]:
    """Trains a Core model.

    Args:
        domain: Path to the domain file.
        config: Path to the config file for Core.
        stories: Path to the Core training data.
        output: Output path.
        fixed_model_name: Name of model to be stored.
        additional_arguments: Additional training parameters.
        model_to_finetune: Optional path to a model which should be finetuned or
            a directory in case the latest trained model should be used.
        finetuning_epoch_fraction: The fraction currently specified training epochs
            in the model configuration which should be used for finetuning.

    Returns:
        Path to the model archive.

    """
    file_importer = TrainingDataImporter.load_core_importer_from_config(
        config, domain, [stories], additional_arguments
    )
    stories_data = file_importer.get_stories()
    nlu_data = file_importer.get_nlu_data()
    domain = file_importer.get_domain()

    if nlu_data.has_e2e_examples():
        structlogger.error(
            "model_training.train_core.e2e_stories_found",
            event_info=(
                "Stories file contains e2e stories. "
                "Please train using `rasa train` so that "
                "the NLU model is also trained."
            ),
        )
        return None

    if domain.is_empty():
        structlogger.error(
            "model_training.train_core.domain_not_found",
            event_info=(
                "Core training was skipped because no valid "
                "domain file was found. Please specify a valid "
                "domain using '--domain' argument or check "
                "if the provided domain file exists."
            ),
        )
        return None

    if not stories_data:
        structlogger.error(
            "model_training.train_core.stories_not_found",
            event_info=(
                "No stories given. Please provide stories in order to "
                "train a Rasa Core model using the '--stories' argument."
            ),
        )
        return None

    _check_unresolved_slots(domain, stories_data)

    return (
        await _train_graph(
            file_importer,
            training_type=TrainingType.CORE,
            output_path=output,
            model_to_finetune=model_to_finetune,
            fixed_model_name=fixed_model_name,
            finetuning_epoch_fraction=finetuning_epoch_fraction,
            **(additional_arguments or {}),
        )
    ).model


async def train_nlu(
    config: Text,
    nlu_data: Optional[Text],
    output: Text,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    additional_arguments: Optional[Dict] = None,
    domain: Optional[Union[Domain, Text]] = None,
    model_to_finetune: Optional[Text] = None,
    finetuning_epoch_fraction: float = 1.0,
) -> Optional[Text]:
    """Trains an NLU model.

    Args:
        config: Path to the config file for NLU.
        nlu_data: Path to the NLU training data.
        output: Output path.
        fixed_model_name: Name of the model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
                                   with the model.
        additional_arguments: Additional training parameters which will be passed to
                              the `train` method of each component.
        domain: Path to the optional domain file/Domain object.
        model_to_finetune: Optional path to a model which should be finetuned or
            a directory in case the latest trained model should be used.
        finetuning_epoch_fraction: The fraction currently specified training epochs
            in the model configuration which should be used for finetuning.

    Returns:
        Path to the model archive.
    """
    if not nlu_data:
        structlogger.error(
            "model_training.train_nlu.nlu_data_not_found",
            event_info=(
                "No NLU data given. Please provide NLU data in order to train "
                "a Rasa NLU model using the '--nlu' argument."
            ),
        )
        return None

    # training NLU only hence the training files still have to be selected
    file_importer = TrainingDataImporter.load_nlu_importer_from_config(
        config, domain, training_data_paths=[nlu_data], args=additional_arguments
    )

    training_data = file_importer.get_nlu_data()
    if training_data.contains_no_pure_nlu_data():
        structlogger.error(
            "model_training.train_nlu.nlu_data_invalid",
            path=nlu_data,
            event_info=(
                f"Path '{nlu_data}' doesn't contain valid NLU data in it. "
                f"Please verify the data format. "
                f"The NLU model training will be skipped now."
            ),
        )
        return None

    return (
        await _train_graph(
            file_importer,
            training_type=TrainingType.NLU,
            output_path=output,
            model_to_finetune=model_to_finetune,
            fixed_model_name=fixed_model_name,
            finetuning_epoch_fraction=finetuning_epoch_fraction,
            persist_nlu_training_data=persist_nlu_training_data,
            **(additional_arguments or {}),
        )
    ).model


def push_model_to_remote_storage(model_path: Path, remote_storage: StorageType) -> None:
    """push model to remote storage"""
    from rasa.nlu.persistor import get_persistor

    persistor = get_persistor(remote_storage)

    if persistor is not None:
        persistor.persist(str(model_path))

    else:
        raise RasaException(
            f"Persistor not found for remote storage: '{remote_storage}'."
        )
