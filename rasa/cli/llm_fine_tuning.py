import argparse
import asyncio
import sys
from typing import List, Any, Dict

import structlog

import rasa.cli.utils
import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.shared.utils.yaml
from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import (
    add_endpoint_param,
    add_model_param,
    add_remote_storage_param,
)
from rasa.cli.e2e_test import (
    read_test_cases,
    validate_model_path,
    RASA_PRO_BETA_FINE_TUNING_RECIPE_ENV_VAR_NAME,
)
from rasa.core.exceptions import AgentNotReady
from rasa.core.utils import AvailableEndpoints
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.llm_fine_tuning.annotation_module import annotate_e2e_tests
from rasa.llm_fine_tuning.llm_data_preparation_module import convert_to_fine_tuning_data
from rasa.llm_fine_tuning.paraphrasing.conversation_rephraser import (
    ConversationRephraser,
)
from rasa.llm_fine_tuning.paraphrasing_module import create_paraphrased_conversations
from rasa.llm_fine_tuning.storage import (
    StorageContext,
    StorageType,
    FileStorageStrategy,
)
from rasa.llm_fine_tuning.train_test_split_module import (
    split_llm_fine_tuning_data,
    INSTRUCTION_DATA_FORMAT,
    CONVERSATIONAL_DATA_FORMAT,
)
from rasa.shared.constants import (
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODELS_PATH,
    LLM_CONFIG_KEY,
)
from rasa.shared.utils.yaml import read_config_file
from rasa.utils.beta import ensure_beta_feature_is_enabled

DEFAULT_INPUT_E2E_TEST_PATH = "e2e_tests"
DEFAULT_OUTPUT_FOLDER = "output"
RESULT_SUMMARY_FILE = "result_summary.yaml"
PARAMETERS_FILE = "params.yaml"

structlogger = structlog.get_logger()


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add the llm fine-tuning subparser to `rasa test`.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    llm_parser = subparsers.add_parser(
        "llm",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Commands related to LLMs.",
    )
    llm_subparsers = llm_parser.add_subparsers()

    llm_finetune_parser = llm_subparsers.add_parser(
        "finetune",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Commands related to LLM fine-tuning.",
    )
    llm_finetune_subparser = llm_finetune_parser.add_subparsers()

    create_llm_finetune_data_preparation_subparser(llm_finetune_subparser, parents)


def create_llm_finetune_data_preparation_subparser(
    fine_tune_llm_parser: SubParsersAction,
    parents: List[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Create fine-tuning LLM data preparation subparser."""
    data_preparation_subparser = fine_tune_llm_parser.add_parser(
        "prepare-data",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prepares data for LLM fine-tuning.",
    )

    data_preparation_subparser.set_defaults(func=prepare_llm_fine_tuning_data)

    add_data_preparation_arguments(data_preparation_subparser)
    add_model_param(data_preparation_subparser, add_positional_arg=False)
    add_endpoint_param(
        data_preparation_subparser,
        help_text="Configuration file for the model server and the connectors as a "
        "yml file.",
    )

    return data_preparation_subparser


def add_data_preparation_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for preparing LLM fine-tuning data."""
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help="The output folder to store the data to.",
    )
    parser.add_argument(
        "path-to-e2e-test-cases",
        nargs="?",
        type=str,
        help="Input file or folder containing end-to-end test cases.",
        default=DEFAULT_INPUT_E2E_TEST_PATH,
    )

    add_remote_storage_param(parser)

    rephrasing_arguments = parser.add_argument_group("Rephrasing Module")
    rephrasing_arguments.add_argument(
        "--num-rephrases",
        choices=range(0, 50),
        type=int,
        default=10,
        help="Number of rephrases to be generated per user utterance.",
    )
    rephrasing_arguments.add_argument(
        "--rephrase-config",
        type=str,
        default=None,
        help="Path to config file that contains the configuration of the "
        "rephrasing module.",
    )

    train_test_split_arguments = parser.add_argument_group("Train/Test Split Module")
    train_test_split_arguments.add_argument(
        "--train-frac",
        type=restricted_float,
        default=0.8,
        help="The amount of data that should go into the training dataset. The value "
        "should be >0.0 and <=1.0.",
    )
    train_test_split_arguments.add_argument(
        "--output-format",
        choices=[INSTRUCTION_DATA_FORMAT, CONVERSATIONAL_DATA_FORMAT],
        type=str,
        nargs="?",
        default=INSTRUCTION_DATA_FORMAT,
        help="Format of the output file.",
    )


def prepare_llm_fine_tuning_data(args: argparse.Namespace) -> None:
    """Prepare LLM fine-tuning data.

    Args:
        args: Commandline arguments.
    """
    ensure_beta_feature_is_enabled(
        "LLM fine-tuning recipe",
        env_flag=RASA_PRO_BETA_FINE_TUNING_RECIPE_ENV_VAR_NAME,
    )

    rephrase_config = (
        read_config_file(args.rephrase_config) if args.rephrase_config else {}
    )
    ConversationRephraser.validate_config(rephrase_config)

    # make sure the output directory exists
    output_dir = args.out
    rasa.shared.utils.io.create_directory(output_dir)

    # read e2e test cases
    path_to_test_cases = getattr(
        args, "path-to-e2e-test-cases", DEFAULT_INPUT_E2E_TEST_PATH
    )
    test_suite = read_test_cases(path_to_test_cases)
    # set up the e2e test runner
    e2e_test_runner = set_up_e2e_test_runner(args)

    if e2e_test_runner.agent.processor is None:
        rasa.shared.utils.cli.print_error(
            "No processor: Not able to retrieve flows and config from trained model."
        )
        sys.exit(0)

    flows = asyncio.run(e2e_test_runner.agent.processor.get_flows())
    llm_command_generator_config = _get_llm_command_generator_config(e2e_test_runner)

    # set up storage context
    storage_context = create_storage_context(StorageType.FILE, output_dir)

    statistics = {}

    # 1. annotate e2e tests
    log_start_of_module("Annotation")
    conversations = annotate_e2e_tests(e2e_test_runner, test_suite, storage_context)
    statistics["num_input_e2e_tests"] = len(test_suite.test_cases)
    statistics["num_annotated_conversations"] = len(conversations)
    statistics["num_user_messages_across_conversations"] = sum(
        [len(conversation.get_user_messages()) for conversation in conversations]
    )
    statistics["num_user_messages_to_rephrase_across_conversations"] = sum(
        [
            len(conversation.get_user_messages_to_rephrase())
            for conversation in conversations
        ]
    )
    log_end_of_module("Annotation", statistics)

    # 2. paraphrase conversations
    log_start_of_module("Rephrasing")
    conversations, rephrase_config = asyncio.run(
        create_paraphrased_conversations(
            conversations,
            rephrase_config,
            args.num_rephrases,
            flows,
            llm_command_generator_config,
            storage_context,
        )
    )
    statistics["num_passing_rephrased_user_messages"] = sum(
        [conversation.get_number_of_rephrases(True) for conversation in conversations]
    )
    statistics["num_failing_rephrased_user_messages"] = sum(
        [conversation.get_number_of_rephrases(False) for conversation in conversations]
    )
    log_end_of_module("Rephrasing", statistics)

    # 3. create fine-tuning dataset
    log_start_of_module("LLM Data Preparation")
    llm_fine_tuning_data = convert_to_fine_tuning_data(conversations, storage_context)
    statistics["num_ft_data_points"] = len(llm_fine_tuning_data)
    log_end_of_module("LLM Data Preparation", statistics)

    # 4. create train/test split
    log_start_of_module("Train/Test Split")
    train_data, val_data = split_llm_fine_tuning_data(
        llm_fine_tuning_data,
        args.train_frac,
        args.output_format,
        storage_context,
        test_suite,
    )
    statistics["num_train_data_points"] = len(train_data)
    statistics["num_val_data_points"] = len(val_data)
    log_end_of_module("Train/Test Split", statistics)

    # write down params and statistics to a file
    write_params(args, rephrase_config, output_dir)
    write_statistics(statistics, output_dir)

    rasa.shared.utils.cli.print_success(
        f"Data and intermediate results are written " f"to '{output_dir}'."
    )


def _get_llm_command_generator_config(e2e_test_runner: E2ETestRunner) -> Dict[str, Any]:
    from rasa.dialogue_understanding.generator.constants import DEFAULT_LLM_CONFIG

    train_schema = e2e_test_runner.agent.processor.model_metadata.train_schema  # type: ignore

    for node in train_schema.nodes:
        if "SingleStepLLMCommandGenerator" in node:
            return {
                **DEFAULT_LLM_CONFIG,
                **train_schema.nodes[node].config.get(LLM_CONFIG_KEY),
            }

    rasa.shared.utils.cli.print_error(
        "The provided model was not trained with the 'SingleStepLLMCommandGenerator'."
        "Without the 'SingleStepLLMCommandGenerator' no data for fine-tuning can be "
        "created. Please add the 'SingleStepLLMCommandGenerator' to your config and"
        "train your model."
    )
    sys.exit(0)


def log_start_of_module(module_name: str) -> None:
    log_info = f"Starting {module_name} Module"
    rasa.shared.utils.cli.print_info(
        f"{rasa.shared.utils.cli.pad(log_info, char='-')}\n"
    )


def log_end_of_module(module_name: str, statistics: Dict[str, int]) -> None:
    log_info = f"Finished {module_name} Module"
    rasa.shared.utils.cli.print_info(
        f"{rasa.shared.utils.cli.pad(log_info, char='-')}\n"
    )
    rasa.shared.utils.cli.print_color(
        "Current Statistics:", color=rasa.shared.utils.io.bcolors.BOLD
    )
    for key, value in statistics.items():
        rasa.shared.utils.cli.print_color(
            f"  {key}: {value}", color=rasa.shared.utils.io.bcolors.BOLD
        )


def restricted_float(x: Any) -> float:
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def write_params(
    args: argparse.Namespace, rephrase_config: Dict[str, Any], output_path: str
) -> None:
    yaml_data = {
        "parameters": {
            "num_rephrases": args.num_rephrases,
            "rephrase_config": rephrase_config,
            "model": args.model,
            "endpoints": args.endpoints,
            "remote-storage": args.remote_storage,
            "train_frac": args.train_frac,
            "output_format": args.output_format,
            "out": output_path,
        }
    }

    rasa.shared.utils.yaml.write_yaml(yaml_data, f"{output_path}/{PARAMETERS_FILE}")


def write_statistics(statistics: Dict[str, Any], output_path: str) -> None:
    rasa.shared.utils.yaml.write_yaml(
        statistics, f"{output_path}/{RESULT_SUMMARY_FILE}"
    )


def get_valid_endpoints(endpoints_file: str) -> AvailableEndpoints:
    validated_endpoints_file = rasa.cli.utils.get_validated_path(
        endpoints_file, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    endpoints = AvailableEndpoints.read_endpoints(validated_endpoints_file)

    # Ignore all endpoints apart from action server, model, nlu and nlg
    # to ensure InMemoryTrackerStore is being used instead of production
    # tracker store
    endpoints.tracker_store = None
    endpoints.lock_store = None
    endpoints.event_broker = None

    return endpoints


def set_up_e2e_test_runner(args: argparse.Namespace) -> E2ETestRunner:
    endpoints = get_valid_endpoints(args.endpoints)

    if endpoints.model is None:
        args.model = validate_model_path(args.model, "model", DEFAULT_MODELS_PATH)

    try:
        return E2ETestRunner(
            remote_storage=args.remote_storage,
            model_path=args.model,
            model_server=endpoints.model,
            endpoints=endpoints,
        )
    except AgentNotReady as error:
        structlogger.error(
            "cli.finetune_llm.prepare_data.set_up_e2e_test_runner", error=error.message
        )
        sys.exit(1)


def create_storage_context(
    storage_type: StorageType, output_dir: str
) -> StorageContext:
    if storage_type == StorageType.FILE:
        strategy = FileStorageStrategy(output_dir)
    else:
        raise ValueError("Unsupported storage type")

    return StorageContext(strategy)
