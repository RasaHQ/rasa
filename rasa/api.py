import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text, Union

import rasa.shared.constants
from rasa.core.config.configuration import Configuration
from rasa.core.persistor import StorageType

# WARNING: Be careful about adding any top level imports at this place!
#   These functions are imported in `rasa.__init__` and any top level import
#   added here will get executed as soon as someone runs `import rasa`.
#   Some imports are very slow (e.g. `tensorflow`) and we want them to get
#   imported when running `import rasa`. If you add more imports here,
#   please check that in the chain you are importing, no slow packages
#   are getting imported.

if TYPE_CHECKING:
    from rasa.model_training import TrainingResult
    from rasa.shared.importers.importer import TrainingDataImporter


def run(
    model: Text,
    sub_agents: Text,
    connector: Optional[Text] = None,
    **kwargs: Dict[Text, Any],
) -> None:
    """Runs a Rasa model.

    Args:
        model: Path to model archive.
        sub_agents: Path to sub-agents directory.
        connector: Connector which should be use (overwrites `credentials`
        field).
        **kwargs: Additional arguments which are passed to
        `rasa.core.run.serve_application`.

    """
    import rasa.core.run
    import rasa.shared.utils.common
    from rasa.shared.constants import DOCS_BASE_URL
    from rasa.shared.utils.cli import print_warning

    _sub_agents = Configuration.get_instance().available_agents

    credentials = Configuration.get_instance().credentials

    if not connector and not credentials:
        connector = "rest"

        print_warning(
            f"No chat connector configured, falling back to the "
            f"REST input channel. To connect your bot to another channel, "
            f"read the docs here: {DOCS_BASE_URL}/messaging-and-voice-channels"
        )

    kwargs = rasa.shared.utils.common.minimal_kwargs(
        kwargs, rasa.core.run.serve_application
    )

    if "endpoints" in kwargs:
        del kwargs["endpoints"]

    if "credentials" in kwargs:
        del kwargs["credentials"]

    rasa.core.run.serve_application(
        model,
        channel=connector,
        credentials=credentials,
        endpoints=Configuration.get_instance().endpoints,
        sub_agents=_sub_agents,
        **kwargs,
    )


def train(
    domain: Text,
    config: Text,
    training_files: "Union[Text, List[Text]]",
    endpoints: Text = rasa.shared.constants.DEFAULT_ENDPOINTS_PATH,
    output: Text = rasa.shared.constants.DEFAULT_MODELS_PATH,
    dry_run: bool = False,
    force_training: bool = False,
    fixed_model_name: Optional[str] = None,
    persist_nlu_training_data: bool = False,
    core_additional_arguments: Optional[Dict] = None,
    nlu_additional_arguments: Optional[Dict] = None,
    model_to_finetune: Optional[str] = None,
    finetuning_epoch_fraction: float = 1.0,
    remote_storage: Optional[StorageType] = None,
    file_importer: Optional["TrainingDataImporter"] = None,
    keep_local_model_copy: bool = False,
    remote_root_only: bool = False,
    sub_agents: Optional[str] = None,
) -> "TrainingResult":
    """Runs Rasa Core and NLU training in `async` loop.

    Args:
        domain: Path to the domain file.
        config: Path to the config for Core and NLU.
        endpoints: Path to the endpoints file.
        training_files: Paths to the training data for Core and NLU.
        output: Output path.
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
        remote_storage: Optional name of the remote storage to
            use for storing the model.
        file_importer: Instance of `TrainingDataImporter` to use for training.
            If it is not provided, a new instance will be created.
        keep_local_model_copy: If `True` the model will be stored locally even if
            remote storage is configured.
        remote_root_only: If `True`, the model will be stored in the root of the
            remote model storage.
        sub_agents: Path to sub-agents directory.

    Returns:
        An instance of `TrainingResult`.
    """
    from rasa.model_training import train

    return asyncio.run(
        train(
            domain=domain,
            config=config,
            training_files=training_files,
            endpoints=endpoints,
            output=output,
            dry_run=dry_run,
            force_training=force_training,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
            core_additional_arguments=core_additional_arguments,
            nlu_additional_arguments=nlu_additional_arguments,
            model_to_finetune=model_to_finetune,
            finetuning_epoch_fraction=finetuning_epoch_fraction,
            remote_storage=remote_storage,
            file_importer=file_importer,
            keep_local_model_copy=keep_local_model_copy,
            remote_root_only=remote_root_only,
            sub_agents=sub_agents,
        )
    )


def test(
    model: Text,
    stories: Text,
    nlu_data: Text,
    output: Text = rasa.shared.constants.DEFAULT_RESULTS_PATH,
    additional_arguments: Optional[Dict] = None,
) -> None:
    """Test a Rasa model against a set of test data.

    Args:
        model: model to test
        stories: path to the dialogue test data
        nlu_data: path to the NLU test data
        output: path to folder where all output will be stored
        additional_arguments: additional arguments for the test call
    """
    from rasa.model_testing import test_core, test_nlu

    if additional_arguments is None:
        additional_arguments = {}

    test_core(model, stories, output, additional_arguments)  # type: ignore[unused-coroutine]
    test_nlu(model, nlu_data, output, additional_arguments)  # type: ignore[unused-coroutine]
