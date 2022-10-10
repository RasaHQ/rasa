import os
from typing import Text, Union

from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.utils.cli import print_error
import rasa.shared.nlu.training_data.loading
from rasa.nlu.utils import write_to_file


def convert_training_data(
    data_file: Union[list, Text], out_file: Text, output_format: Text, language: Text
) -> None:
    """Convert training data.

    Args:
        data_file (Union[list, Text]): Path to the file or directory
            containing Rasa data.
        out_file (Text): File or existing path where to save
            training data in Rasa format.
        output_format (Text): Output format the training data
            should be converted into.
        language (Text): Language of the data.
    """
    if isinstance(data_file, list):
        data_file = data_file[0]

    if not os.path.exists(str(data_file)):
        print_error(
            "Data file '{}' does not exist. Provide a valid NLU data file using "
            "the '--data' argument.".format(data_file)
        )
        return

    td = rasa.shared.nlu.training_data.loading.load_data(data_file, language)
    if output_format == "json":
        output = td.nlu_as_json(indent=2)
    elif output_format == "md":
        output = td.nlu_as_markdown()
    else:
        output = RasaYAMLWriter().dumps(td)

    write_to_file(out_file, output)
