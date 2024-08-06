import argparse
import asyncio
import importlib
import re
from csv import Error as CSVError
from pathlib import Path
from textwrap import dedent
from typing import Text, Dict, List, Optional, Any

import pandas as pd
import structlog
from jinja2 import Template
from ruamel import yaml

from rasa.exceptions import RasaException
from rasa.shared.utils.cli import print_error_and_exit
from rasa.shared.utils.llm import llm_factory

structlogger = structlog.get_logger()

DEFAULT_E2E_TEST_GENERATOR_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.e2e_test", "e2e_test_converter_prompt.jinja2"
)

CSV = ".csv"
XLSX = ".xlsx"
XLS = ".xls"
EXCEL_EXTENSIONS = [XLS, XLSX]
ALLOWED_EXTENSIONS = [CSV, *EXCEL_EXTENSIONS]

NUMBER_OF_LLM_ATTEMPTS = 3
DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 60,
    "max_tokens": 2048,
    "model_name": "gpt-4o-mini",
}


class E2ETestConverter:
    """E2ETestConvertor class is responsible for reading input CSV or XLS/XLSX files,
    splitting the data into distinct conversations, converting them into test cases,
    and storing the test cases into a YAML file at a specified directory.
    """

    def __init__(
        self,
        path: Text,
        sheet_name: Optional[Text] = None,
        prompt_template: Text = DEFAULT_E2E_TEST_GENERATOR_PROMPT_TEMPLATE,
        **kwargs: Any,
    ) -> None:
        """Initializes the E2ETestConverter with necessary parameters.

        Args:
            path (Text): Path to the input file.
            sheet_name (Text): Name of the sheet in XLSX file.
            prompt_template (Text): Path to the jinja2 template.
        """
        self.input_path: Text = path
        self.sheet_name: Optional[Text] = sheet_name
        self.prompt_template: Text = prompt_template

    @staticmethod
    def is_yaml_valid(yaml_string: Text) -> bool:
        """Validate the string against the YAML format.

        Args:
            yaml_string (Text): String to be validated

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            yaml.safe_load(yaml_string)
            return True
        except Exception as e:
            structlogger.debug("e2e_test_generator.yaml_string_invalid", e=e)
            return False

    @staticmethod
    def remove_markdown_code_syntax(markdown_string: Text) -> Text:
        """
        Remove Markdown code formatting from the string.

        Args:
            markdown_string (Text): string to be parsed.

        Returns:
            Text: Parsed string.

        Example:
            >>> markdown_string = "```yaml\nkey: value\n```"
            >>> remove_markdown_formatting(markdown_string)
            'key: value'
        """
        if not markdown_string:
            return ""

        dedented_string = dedent(markdown_string)

        # Updated regex to handle optional language identifier after opening triple backticks.
        regex = r"^```.*\n|```$"
        return re.sub(regex, "", dedented_string, flags=re.MULTILINE).strip()

    @staticmethod
    async def generate_llm_response(prompt: Text) -> Optional[Text]:
        """Use LLM to generate a response.

        Args:
            prompt (Text): the prompt to send to the LLM.

        Returns:
            Optional[Text]: Generated response.
        """
        llm = llm_factory(DEFAULT_LLM_CONFIG, DEFAULT_LLM_CONFIG)

        try:
            return await llm.apredict(prompt)
        except Exception as exc:
            structlogger.debug("e2e_test_generator.llm_response_error", exc=exc)
            return None

    @staticmethod
    def split_data_into_conversations(
        input_data: List[Dict],
    ) -> List[List[Dict[Text, Any]]]:
        """Splits the data into conversations using empty row as a separator.

        Arguments:
            input_data (List[Dict]): The list of rows of the input file.

        Returns:
            List[List[Dict[Text, Any]]]: List of conversations
        """
        conversations, conversation = [], []

        for row in input_data:
            # Remove empty values from the row
            row = {key: value for key, value in row.items() if value}

            # Iterate through each row until the empty row separator is hit
            if row:
                conversation.append(row)
            else:
                # If the current conversation exists, add it to the list
                if conversation:
                    conversations.append(conversation)
                    conversation = []

        # Add last conversation to the list as it might not have a separator after it
        if conversation:
            conversations.append(conversation)

        return conversations

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

    def read_file(self) -> List[Dict]:
        """Calls the appropriate file reading method based on the file extension.

        Raises:
            RasaException: If the file could not be read.

        Returns:
            List[Dict]: Parsed data from the input file as a list of dictionaries.
        """
        extension_to_method = {
            CSV: self.read_csv,
            XLS: self.read_xlsx,
            XLSX: self.read_xlsx,
        }

        input_file_extension = self.get_and_validate_input_file_extension()

        try:
            structlogger.debug(
                "e2e_test_generator.read_file",
                input_file_extension=input_file_extension,
            )
            return extension_to_method[input_file_extension]()
        except pd.errors.ParserError:
            raise RasaException("The file could not be read due to a parsing error.")
        except pd.errors.EmptyDataError:
            raise RasaException("The file is empty and could not be read.")
        except CSVError:
            raise RasaException("There was an error with reading the CSV file.")
        except ValueError:
            raise RasaException("There was a value error while reading the file.")

    def render_template(self, conversation: List[Dict[Text, Any]]) -> Text:
        """Renders a jinja2 template.

        Arguments:
            conversation (List[Dict[Text, Any]]): The list of rows of a conversation.

        Returns:
            Text: The rendered template string.
        """
        kwargs = {"conversation": conversation}
        return Template(self.prompt_template).render(**kwargs)

    async def convert_single_conversation_into_test(
        self, conversation: List[Dict[Text, Any]]
    ) -> Text:
        """Uses LLM to convert a conversation to a YAML test case.

        Arguments:
            conversation (List[Dict[Text, Any]]): The list of rows of a conversation.

        Returns:
            Text: YAML representation of the conversation.
        """
        yaml_test_case = ""
        prompt_template = self.render_template(conversation)

        for _ in range(NUMBER_OF_LLM_ATTEMPTS):
            response = await self.generate_llm_response(prompt_template)
            yaml_test_case = self.remove_markdown_code_syntax(response)
            if not self.is_yaml_valid(yaml_test_case):
                structlogger.debug("e2e_test_generator.invalid_yaml_format")
                continue
            break

        return yaml_test_case

    async def convert_conversations_into_tests(
        self, conversations: List[List[Dict[Text, Any]]]
    ) -> Text:
        """Generates test cases from the parsed data.

        Arguments:
            conversations (List[List[Dict[Text, Any]]]): The list of conversation rows.

        Returns:
            Text: YAML representation of the test cases.
        """
        # Convert all conversations into YAML test cases asynchronously.
        tasks = [
            asyncio.ensure_future(
                self.convert_single_conversation_into_test(conversation)
            )
            for conversation in conversations
        ]
        results = await asyncio.gather(*tasks)

        structlogger.debug("e2e_test_generator.test_generation_finished")
        return "\n".join(results)

    async def run(self) -> None:
        """Executes the E2E test conversion process: reads the file, generates tests,
        and writes them to a YAML file.
        """
        input_data = self.read_file()
        conversations = self.split_data_into_conversations(input_data)
        yaml_tests_string = await self.convert_conversations_into_tests(conversations)


def convert_data_to_e2e_tests(args: argparse.Namespace) -> None:
    converter = E2ETestConverter(**vars(args))
    try:
        asyncio.run(converter.run())
    except RasaException as exc:
        structlogger.error("e2e_test_converter.failed.run", exc=exc)
        print_error_and_exit(f"Failed to convert the data into E2E tests. Error: {exc}")
