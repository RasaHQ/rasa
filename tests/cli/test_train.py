from typing import Callable

from pytest import RunResult


def test_rasa_train(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa studio train [-h] [-v] [-vv] [--quiet]
                    [--logging-config-file LOGGING_CONFIG_FILE]
                    [--data DATA [DATA ...]] [-c CONFIG]
                    [-d DOMAIN] [--out OUT] [--dry-run]
                    [--augmentation AUGMENTATION]
                    [--debug-plots] [--num-threads NUM_THREADS]
                    [--fixed-model-name FIXED_MODEL_NAME]
                    [--persist-nlu-data] [--force]
                    [--finetune [FINETUNE]]
                    [--epoch-fraction EPOCH_FRACTION]
                    [--endpoints ENDPOINTS]
                    [--entities ENTITIES [ENTITIES ...]]
                    [--intents INTENTS [INTENTS ...]]
                    assistant_name"""
    lines = help_text.split("\n")

    output = run("studio", "train", "--help")

    printed_help = [line.strip() for line in output.outlines]
    printed_help = str.join(" ", printed_help)  # type: ignore
    for line in lines:
        assert line.strip() in printed_help
