import argparse
import os
from pathlib import Path
from typing import Text, Optional, List, Dict, Any

from rasa.cli.utils import print_error, print_success

from rasa.nlu import training_data
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.training_data.formats import MarkdownReader, NLGMarkdownReader
from rasa.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.nlu.utils import write_to_file
from rasa.nlu.utils.pattern_utils import read_lookup_table_file
from rasa.utils.io import generate_path_for_converted_training_data_file


def convert_training_data(
    data_file: Text, out_file: Text, output_format: Text, language: Text
):
    if not os.path.exists(data_file):
        print_error(
            "Data file '{}' does not exist. Provide a valid NLU data file using "
            "the '--data' argument.".format(data_file)
        )
        return

    if output_format == "json":
        td = training_data.load_data(data_file, language)
        output = td.nlu_as_json(indent=2)
    elif output_format == "md":
        td = training_data.load_data(data_file, language)
        output = td.nlu_as_markdown()
    else:
        print_error(
            "Did not recognize output format. Supported output formats: 'json' and "
            "'md'. Specify the desired output format with '--format'."
        )
        return

    write_to_file(out_file, output)


def write_nlu_yaml(training_data_path: Path, output_dir_path: Path) -> None:
    """Converts and writes NLU training data from `Markdown` to `YAML` format.

        Args:
            training_data_path: Path to the markdown file.
            output_dir_path: Path to the target output directory.
    """
    output_nlu_path = generate_path_for_converted_training_data_file(
        training_data_path, output_dir_path
    )

    yaml_training_data = MarkdownReader().read(training_data_path)
    RasaYAMLWriter().dump(output_nlu_path, yaml_training_data)

    for lookup_table in yaml_training_data.lookup_tables:
        write_nlu_lookup_table_yaml(lookup_table, output_dir_path)

    print_success(f"Converted NLU file: '{training_data_path}' >> '{output_nlu_path}'.")


def write_nlu_lookup_table_yaml(
    lookup_table: Dict[Text, Any], output_dir_path: Path
) -> None:
    """Converts and writes lookup tables examples from `txt` to `YAML` format.

    Args:
        lookup_table: Lookup tables items.
        output_dir_path: Path to the target output directory.
    """
    lookup_table_file = lookup_table.get("elements")
    if not lookup_table_file or not isinstance(lookup_table_file, str):
        return

    examples_from_file = read_lookup_table_file(lookup_table_file)
    target_filename = generate_path_for_converted_training_data_file(
        Path(lookup_table_file), output_dir_path
    )
    entity_name = Path(lookup_table_file).stem

    RasaYAMLWriter().dump(
        target_filename,
        TrainingData(
            lookup_tables=[{"name": entity_name, "elements": examples_from_file}]
        ),
    )


def write_nlg_yaml(training_data_path: Path, output_dir_path: Path) -> None:
    """Converts and writes NLG (retrieval intents) training data
    from `Markdown` to `YAML` format.

    Args:
        training_data_path: Path to the markdown file.
        output_dir_path: Path to the target output directory.
    """
    reader = NLGMarkdownReader()
    writer = RasaYAMLWriter()

    output_nlg_path = generate_path_for_converted_training_data_file(
        training_data_path, output_dir_path
    )

    yaml_training_data = reader.read(training_data_path)
    writer.dump(output_nlg_path, yaml_training_data)

    print_success(f"Converted NLG file: '{training_data_path}' >> '{output_nlg_path}'.")


def main(args: argparse.Namespace):
    convert_training_data(args.data, args.out, args.format, args.language)


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.convert` directly is "
        "no longer supported. "
        "Please use `rasa data` instead."
    )
