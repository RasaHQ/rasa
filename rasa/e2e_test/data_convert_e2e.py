import argparse
from csv import Error as CSVError
from pathlib import Path
from typing import Text, Dict, List, Optional, Any
from rasa.shared.utils.cli import print_error_and_exit
import pandas as pd
import structlog

from rasa.exceptions import RasaException

structlogger = structlog.get_logger()


CSV = ".csv"
XLSX = ".xlsx"
XLS = ".xls"
EXCEL_EXTENSIONS = [XLS, XLSX]
ALLOWED_EXTENSIONS = [CSV, *EXCEL_EXTENSIONS]


class E2ETestConverter:
    """E2ETestConvertor class is responsible for reading input CSV or XLS/XLSX files,
    splitting the data into distinct conversations, converting them into test cases,
    and storing the test cases into a YAML file at a specified directory.

    Attributes:
        input_path (Text): Path to the input file.
        sheet_name (Text): Name of the sheet in XLSX file (if applicable).
        data (List[Dict]): Parsed data from the input file.
    """
    def __init__(
        self,
        path: Text,
        sheet_name: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the E2ETestConverter with necessary parameters.

        Args:
            input_path (Text): Path to the input file.
            sheet_name (Text): Name of the sheet in XLSX file.
        """
        self.input_path: Text = path
        self.sheet_name: Optional[Text] = sheet_name
        self._data: Optional[List[Dict]] = None

    @property
    def data(self) -> Optional[List[Dict]]:
        """Getter for the data attribute.

        Returns:
            Optional[List[Dict]]: Parsed data from the input file.
        """
        return self._data

    @data.setter
    def data(self, value: List[Dict]) -> None:
        """Setter for the data attribute.

        Args:
            value (List[Dict]): Data to be set.
        """
        self._data = value

    def get_and_validate_input_file_extension(self) -> Text:
        """Validates the input file extension and checks for required properties
        like sheet name for XLSX files.

        Raises:
            RasaException: If the directory path is provided, file extension
            is not supported or sheet name is missing.
        """
        input_file_extension = Path(self.input_path).suffix.lower()
        if not input_file_extension:
            raise RasaException(
                "The path must point to a specific file, not a directory."
            )

        if input_file_extension not in ALLOWED_EXTENSIONS:
            raise RasaException(
                f"Unsupported file type: {input_file_extension}. "
                "Please use .csv or .xls/.xlsx"
            )

        if input_file_extension in EXCEL_EXTENSIONS and self.sheet_name is None:
            raise RasaException(
                f"Please provide a sheet name for the {input_file_extension} file."
            )

        return input_file_extension

    def read_csv(self) -> List[Dict]:
        """Reads a CSV file and converts it into a list of dictionaries.

        Returns:
            List[Dict]: Parsed data from the CSV file as a list of dictionaries.
        """
        df = pd.read_csv(self.input_path, sep=None, engine="python")
        df = df.fillna("")
        return df.to_dict(orient="records")

    def read_xlsx(self) -> List[Dict]:
        """Reads an XLSX file and converts it into a list of dictionaries.

        Returns:
            List[Dict]: Parsed data from the XLSX file as a list of dictionaries.
        """
        df = pd.read_excel(self.input_path, sheet_name=self.sheet_name)
        df = df.fillna("")
        return df.to_dict(orient="records")

    def read_file(self) -> None:
        """Calls the appropriate file reading method based on the file extension.

        Raises:
            RasaException: If the file could not be read.
        """
        extension_to_method = {
            CSV: self.read_csv,
            XLS: self.read_xlsx,
            XLSX: self.read_xlsx,
        }

        input_file_extension = self.get_and_validate_input_file_extension()

        try:
            self.data = extension_to_method[input_file_extension]()
            structlogger.debug(
                "e2e_test_generator.read_file",
                input_file_extension=input_file_extension,
            )
        except pd.errors.ParserError:
            raise RasaException("The file could not be read due to a parsing error.")
        except pd.errors.EmptyDataError:
            raise RasaException("The file is empty and could not be read.")
        except CSVError:
            raise RasaException("There was an error with reading the CSV file.")
        except ValueError:
            raise RasaException("There was a value error while reading the file.")

    def run(self) -> None:
        """Executes the E2E test conversion process: reads the file, generates tests,
        and writes them to a YAML file.
        """
        self.read_file()


def convert_data_to_e2e_tests(args: argparse.Namespace) -> None:
    converter = E2ETestConverter(**vars(args))
    try:
        converter.run()
    except RasaException as exc:
        structlogger.error("e2e_test_converter.failed.run", exc=exc)
        print_error_and_exit(
            f"Failed to convert the data into E2E tests. " f"Error: {exc}"
        )
