import time
from pathlib import Path
from typing import Text, NamedTuple, Optional, List, Union, Dict, Any

import randomname

import rasa.engine.validation
from rasa.engine.caching import LocalTrainingCache
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.components import FingerprintStatus
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.data import TrainingType
from rasa.shared.importers.importer import TrainingDataImporter
from rasa import telemetry
from rasa.shared.core.domain import Domain
import rasa.shared.utils.common
import rasa.utils.common
import rasa.shared.utils.common
import rasa.shared.utils.cli
import rasa.shared.exceptions
import rasa.shared.utils.io
import rasa.shared.constants
import rasa.model

CODE_NEEDS_TO_BE_RETRAINED = 0b0001
CODE_FORCED_TRAINING = 0b1000


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
        rasa.shared.utils.cli.print_warning("The training was forced.")
        return TrainingResult(
            code=CODE_FORCED_TRAINING, dry_run_results=fingerprint_results
        )

    training_required = any(
        isinstance(result, FingerprintStatus) and not result.is_hit
        for result in fingerprint_results.values()
    )

    if training_required:
        rasa.shared.utils.cli.print_warning("The model needs to be retrained.")
        return TrainingResult(
            code=CODE_NEEDS_TO_BE_RETRAINED, dry_run_results=fingerprint_results
        )

    rasa.shared.utils.cli.print_success(
        "No training of components required "
        "(the responses might still need updating!)."
    )
    return TrainingResult(dry_run_results=fingerprint_results)


def train(
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

    Returns:
        An instance of `TrainingResult`.
    """
    file_importer = TrainingDataImporter.load_from_config(
        config, domain, training_files
    )

    stories = file_importer.get_stories()
    nlu_data = file_importer.get_nlu_data()

    training_type = TrainingType.BOTH

    if nlu_data.has_e2e_examples():
        rasa.shared.utils.common.mark_as_experimental_feature("end-to-end training")
        training_type = TrainingType.END_TO_END

    if stories.is_empty() and nlu_data.contains_no_pure_nlu_data():
        rasa.shared.utils.cli.print_error(
            "No training data given. Please provide stories and NLU data in "
            "order to train a Rasa model using the '--data' argument."
        )
        return TrainingResult(code=1)

    domain_object = file_importer.get_domain()
    if domain_object.is_empty():
        rasa.shared.utils.cli.print_warning(
            "Core training was skipped because no valid domain file was found. "
            "Only an NLU-model was created. Please specify a valid domain using "
            "the '--domain' argument or check if the provided domain file exists."
        )
        training_type = TrainingType.NLU

    elif stories.is_empty():
        rasa.shared.utils.cli.print_warning(
            "No stories present. Just a Rasa NLU model will be trained."
        )
        training_type = TrainingType.NLU

    # We will train nlu if there are any nlu example, including from e2e stories.
    elif nlu_data.contains_no_pure_nlu_data() and not nlu_data.has_e2e_examples():
        rasa.shared.utils.cli.print_warning(
            "No NLU data present. Just a Rasa Core model will be trained."
        )
        training_type = TrainingType.CORE

    with telemetry.track_model_training(file_importer, model_type="rasa"):
        return _train_graph(
            file_importer,
            training_type=training_type,
            output_path=output,
            fixed_model_name=fixed_model_name,
            model_to_finetune=model_to_finetune,
            force_full_training=force_training,
            persist_nlu_training_data=persist_nlu_training_data,
            finetuning_epoch_fraction=finetuning_epoch_fraction,
            dry_run=dry_run,
            **(core_additional_arguments or {}),
            **(nlu_additional_arguments or {}),
        )


def _train_graph(
    file_importer: TrainingDataImporter,
    training_type: TrainingType,
    output_path: Text,
    fixed_model_name: Text,
    model_to_finetune: Optional[Union[Text, Path]] = None,
    force_full_training: bool = False,
    dry_run: bool = False,
    **kwargs: Any,
) -> TrainingResult:
    if model_to_finetune:
        model_to_finetune = rasa.model.get_model_for_finetuning(model_to_finetune)
        if not model_to_finetune:
            rasa.shared.utils.cli.print_error_and_exit(
                f"No model for finetuning found. Please make sure to either "
                f"specify a path to a previous model or to have a finetunable "
                f"model within the directory '{output_path}'."
            )

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
    model_configuration = recipe.graph_config_for_recipe(
        config,
        kwargs,
        training_type=training_type,
        is_finetuning=is_finetuning,
    )
    rasa.engine.validation.validate(model_configuration)

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
            fingerprint_status = trainer.fingerprint(
                model_configuration.train_schema, file_importer
            )
            return _dry_run_result(fingerprint_status, force_full_training)

        model_name = _determine_model_name(fixed_model_name, training_type)
        full_model_path = Path(output_path, model_name)

        with telemetry.track_model_training(
            file_importer, model_type=training_type.model_type
        ):
            trainer.train(
                model_configuration,
                file_importer,
                full_model_path,
                force_retraining=force_full_training,
                is_finetuning=is_finetuning,
            )
            rasa.shared.utils.cli.print_success(
                f"Your Rasa model is trained and saved at '{full_model_path}'."
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
        model_file = Path(fixed_model_name)
        if not model_file.name.endswith(".tar.gz"):
            return model_file.with_suffix(".tar.gz").name

        return fixed_model_name

    prefix = ""
    if training_type in [TrainingType.CORE, TrainingType.NLU]:
        prefix = f"{training_type.model_type}-"

    time_format = "%Y%m%d-%H%M%S"
    return f"{prefix}{time.strftime(time_format)}-{randomname.get_name()}.tar.gz"


def train_core(
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
        config, domain, [stories]
    )
    stories_data = file_importer.get_stories()
    nlu_data = file_importer.get_nlu_data()
    domain = file_importer.get_domain()

    if nlu_data.has_e2e_examples():
        rasa.shared.utils.cli.print_error(
            "Stories file contains e2e stories. Please train using `rasa train` so that"
            " the NLU model is also trained."
        )
        return None

    if domain.is_empty():
        rasa.shared.utils.cli.print_error(
            "Core training was skipped because no valid domain file was found. "
            "Please specify a valid domain using '--domain' argument or check "
            "if the provided domain file exists."
        )
        return None

    if not stories_data:
        rasa.shared.utils.cli.print_error(
            "No stories given. Please provide stories in order to "
            "train a Rasa Core model using the '--stories' argument."
        )
        return None

    return _train_graph(
        file_importer,
        training_type=TrainingType.CORE,
        output_path=output,
        model_to_finetune=model_to_finetune,
        fixed_model_name=fixed_model_name,
        finetuning_epoch_fraction=finetuning_epoch_fraction,
        **(additional_arguments or {}),
    ).model


def train_nlu(
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
        rasa.shared.utils.cli.print_error(
            "No NLU data given. Please provide NLU data in order to train "
            "a Rasa NLU model using the '--nlu' argument."
        )
        return None

    # training NLU only hence the training files still have to be selected
    file_importer = TrainingDataImporter.load_nlu_importer_from_config(
        config, domain, training_data_paths=[nlu_data]
    )

    training_data = file_importer.get_nlu_data()
    if training_data.contains_no_pure_nlu_data():
        rasa.shared.utils.cli.print_error(
            f"Path '{nlu_data}' doesn't contain valid NLU data in it. "
            f"Please verify the data format. "
            f"The NLU model training will be skipped now."
        )
        return None

    return _train_graph(
        file_importer,
        training_type=TrainingType.NLU,
        output_path=output,
        model_to_finetune=model_to_finetune,
        fixed_model_name=fixed_model_name,
        finetuning_epoch_fraction=finetuning_epoch_fraction,
        persist_nlu_training_data=persist_nlu_training_data,
        **(additional_arguments or {}),
    ).model
