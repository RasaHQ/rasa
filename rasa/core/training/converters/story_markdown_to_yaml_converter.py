from pathlib import Path

import rasa.shared.constants
from rasa.shared.core.training_data.story_reader.markdown_story_reader import (
    MarkdownStoryReader,
)
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.utils.cli import print_success, print_warning
from rasa.utils.converter import TrainingDataConverter


class StoryMarkdownToYamlConverter(TrainingDataConverter):
    @classmethod
    def filter(cls, source_path: Path) -> bool:
        """Checks if the given training data file contains Core data in `Markdown`
        format and can be converted to `YAML`.

        Args:
            source_path: Path to the training data file.

        Returns:
            `True` if the given file can be converted, `False` otherwise
        """
        return MarkdownStoryReader.is_stories_file(
            source_path
        ) or MarkdownStoryReader.is_test_stories_file(source_path)

    @classmethod
    async def convert_and_write(cls, source_path: Path, output_path: Path) -> None:
        """Converts the given training data file and saves it to the output directory.

        Args:
            source_path: Path to the training data file.
            output_path: Path to the output directory.
        """
        from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
            KEY_ACTIVE_LOOP,
        )

        # check if source file is test stories file
        if MarkdownStoryReader.is_test_stories_file(source_path):
            reader = MarkdownStoryReader(
                is_used_for_training=False,
                use_e2e=True,
                ignore_deprecation_warning=True,
            )
            output_core_path = cls._generate_path_for_converted_test_data_file(
                source_path, output_path
            )
        else:
            reader = MarkdownStoryReader(
                is_used_for_training=False, ignore_deprecation_warning=True
            )
            output_core_path = cls.generate_path_for_converted_training_data_file(
                source_path, output_path
            )

        steps = reader.read_from_file(source_path)

        if YAMLStoryWriter.stories_contain_loops(steps):
            print_warning(
                f"Training data file '{source_path}' contains forms. "
                f"Any 'form' events will be converted to '{KEY_ACTIVE_LOOP}' events. "
                f"Please note that in order for these stories to work you still "
                f"need the 'FormPolicy' to be active. However the 'FormPolicy' is "
                f"deprecated, please consider switching to the new 'RulePolicy', "
                f"for which you can find the documentation here: "
                f"{rasa.shared.constants.DOCS_URL_RULES}."
            )

        writer = YAMLStoryWriter()
        writer.dump(
            output_core_path,
            steps,
            is_test_story=MarkdownStoryReader.is_test_stories_file(source_path),
        )

        print_success(f"Converted Core file: '{source_path}' >> '{output_core_path}'.")

    @classmethod
    def _generate_path_for_converted_test_data_file(
        cls, source_file_path: Path, output_directory: Path
    ) -> Path:
        """Generates path for a test data file converted to YAML format.

        Args:
            source_file_path: Path to the original file.
            output_directory: Path to the target directory.

        Returns:
            Path to the target converted training data file.
        """
        if cls._has_test_prefix(source_file_path):
            return (
                output_directory
                / f"{source_file_path.stem}{cls.converted_file_suffix()}"
            )
        return (
            output_directory / f"{rasa.shared.constants.TEST_STORIES_FILE_PREFIX}"
            f"{source_file_path.stem}{cls.converted_file_suffix()}"
        )

    @classmethod
    def _has_test_prefix(cls, source_file_path: Path) -> bool:
        """Checks if test data file has test prefix.

        Args:
            source_file_path: Path to the original file.

        Returns:
            `True` if the filename starts with the prefix, `False` otherwise.
        """
        return Path(source_file_path).name.startswith(
            rasa.shared.constants.TEST_STORIES_FILE_PREFIX
        )
