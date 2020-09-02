import asyncio
from pathlib import Path

from rasa.cli.utils import print_warning, print_success
from rasa.constants import DOCS_URL_RULES
from rasa.core.training.story_reader.markdown_story_reader import MarkdownStoryReader
from rasa.core.training.story_writer.yaml_story_writer import YAMLStoryWriter
from rasa.utils.io import generate_path_for_converted_training_data_file


def write_core_yaml(training_data_path: Path, output_dir_path: Path) -> None:
    """Converts and writes Core training data from `Markdown` to `YAML` format

    Args:
        training_data_path: Path to the markdown file.
        output_dir_path: Path to the target output directory.
    """
    from rasa.core.training.story_reader.yaml_story_reader import KEY_ACTIVE_LOOP

    output_core_path = generate_path_for_converted_training_data_file(
        training_data_path, output_dir_path
    )

    reader = MarkdownStoryReader(unfold_or_utterances=False)
    writer = YAMLStoryWriter()

    loop = asyncio.get_event_loop()
    steps = loop.run_until_complete(reader.read_from_file(training_data_path))

    if YAMLStoryWriter.stories_contain_loops(steps):
        print_warning(
            f"Training data file '{training_data_path}' contains forms. "
            f"Any 'form' events will be converted to '{KEY_ACTIVE_LOOP}' events. "
            f"Please note that in order for these stories to work you still "
            f"need the 'FormPolicy' to be active. However the 'FormPolicy' is "
            f"deprecated, please consider switching to the new 'RulePolicy', "
            f"for which you can find the documentation here: {DOCS_URL_RULES}."
        )

    writer.dump(output_core_path, steps)

    print_success(
        f"Converted Core file: '{training_data_path}' >> '{output_core_path}'."
    )
