from pathlib import Path
from typing import Dict, Text, Any

from rasa.shared.utils.cli import print_success
from rasa.nlu.utils.pattern_utils import read_lookup_table_file
from rasa.shared.nlu.training_data.formats import MarkdownReader
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.converter import TrainingDataConverter


class NLUMarkdownToYamlConverter(TrainingDataConverter):
    @classmethod
    def filter(cls, source_path: Path) -> bool:
        """Checks if the given training data file contains NLU data in `Markdown` format
        and can be converted to `YAML`.

        Args:
            source_path: Path to the training data file.

        Returns:
            `True` if the given file can be converted, `False` otherwise
        """
        return MarkdownReader.is_markdown_nlu_file(source_path)

    @classmethod
    async def convert_and_write(cls, source_path: Path, output_path: Path) -> None:
        """Converts the given training data file and saves it to the output directory.

        Args:
            source_path: Path to the training data file.
            output_path: Path to the output directory.
        """
        output_nlu_path = cls.generate_path_for_converted_training_data_file(
            source_path, output_path
        )

        yaml_training_data = MarkdownReader().read(source_path)
        RasaYAMLWriter().dump(output_nlu_path, yaml_training_data)

        for lookup_table in yaml_training_data.lookup_tables:
            cls._write_nlu_lookup_table_yaml(lookup_table, output_path)

        print_success(f"Converted NLU file: '{source_path}' >> '{output_nlu_path}'.")

    @classmethod
    def _write_nlu_lookup_table_yaml(
        cls, lookup_table: Dict[Text, Any], output_dir_path: Path
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
        target_filename = cls.generate_path_for_converted_training_data_file(
            Path(lookup_table_file), output_dir_path
        )
        entity_name = Path(lookup_table_file).stem

        RasaYAMLWriter().dump(
            target_filename,
            TrainingData(
                lookup_tables=[{"name": entity_name, "elements": examples_from_file}]
            ),
        )
