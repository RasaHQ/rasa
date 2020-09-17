from pathlib import Path

from rasa.shared.utils.cli import print_success
from rasa.shared.nlu.training_data.formats import NLGMarkdownReader
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.utils.converter import TrainingDataConverter


class NLGMarkdownToYamlConverter(TrainingDataConverter):
    @classmethod
    def filter(cls, source_path: Path) -> bool:
        """Checks if the given training data file contains NLG data in `Markdown` format
        and can be converted to `YAML`.

        Args:
            source_path: Path to the training data file.

        Returns:
            `True` if the given file can be converted, `False` otherwise
        """
        return NLGMarkdownReader.is_markdown_nlg_file(source_path)

    @classmethod
    async def convert_and_write(cls, source_path: Path, output_path: Path) -> None:
        """Converts the given training data file and saves it to the output directory.

        Args:
            source_path: Path to the training data file.
            output_path: Path to the output directory.
        """
        reader = NLGMarkdownReader()
        writer = RasaYAMLWriter()

        output_nlg_path = cls.generate_path_for_converted_training_data_file(
            source_path, output_path
        )

        yaml_training_data = reader.read(source_path)
        writer.dump(output_nlg_path, yaml_training_data)

        print_success(f"Converted NLG file: '{source_path}' >> '{output_nlg_path}'.")
