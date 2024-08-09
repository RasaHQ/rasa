import asyncio
import importlib
import re
from csv import Error as CSVError
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any
from typing import Optional

import pandas as pd
import ruamel
import structlog
from jinja2 import Template
from ruamel.yaml.scanner import ScannerError

from rasa.e2e_test.constants import KEY_TEST_CASES
from rasa.cli.e2e_test import read_e2e_test_schema
from rasa.exceptions import RasaException
from rasa.shared.utils.llm import llm_factory
from rasa.shared.utils.yaml import (
    validate_yaml_data_using_schema_with_assertions,
    YamlValidationException,
)

structlogger = structlog.get_logger()

DEFAULT_E2E_OUTPUT_TESTS_DIRECTORY = "e2e_tests"
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


@dataclass
class ConversationEntry:
    """Class for storing an entry in a conversation."""

    data: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationEntry":
        return cls(data=data)

    def as_dict(self) -> Dict[str, Any]:
        return self.data


@dataclass
class Conversation:
    """Class for storing a single conversation."""

    entries: List[ConversationEntry] = field(default_factory=list)

    def add_entry(self, entry: ConversationEntry) -> None:
        self.entries.append(entry)

    def as_list(self) -> List[Dict[str, Any]]:
        return [entry.as_dict() for entry in self.entries]


@dataclass
class Conversations:
    """Class for storing multiple conversations."""

    conversations: List[Conversation] = field(default_factory=list)

    def add_conversation(self, conversation: Conversation) -> None:
        self.conversations.append(conversation)

    def as_list(self) -> List[List[Dict[str, Any]]]:
        return [conversation.as_list() for conversation in self.conversations]


class E2ETestConverter:
    """E2ETestConverter class is responsible for reading input CSV or XLS/XLSX files,
    splitting the data into distinct conversations, and converting them into test cases.
    """

    def __init__(
        self,
        path: str,
        sheet_name: Optional[str] = None,
        prompt_template: str = DEFAULT_E2E_TEST_GENERATOR_PROMPT_TEMPLATE,
        **kwargs: Any,
    ) -> None:
        """Initializes the E2ETestConverter with necessary parameters.

        Args:
            path (str): Path to the input file.
            sheet_name (str): Name of the sheet in XLSX file.
            prompt_template (str): Path to the jinja2 template.
        """
        self.input_path: str = path
        self.sheet_name: Optional[str] = sheet_name
        self.prompt_template: str = prompt_template

    @staticmethod
    def is_yaml_valid(yaml_string: str) -> bool:
        """Validate the string against the YAML format.

        Args:
            yaml_string (str): String to be validated

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            e2e_schema = read_e2e_test_schema()
            yaml_data = ruamel.yaml.safe_load(yaml_string)
            validate_yaml_data_using_schema_with_assertions(
                yaml_data={KEY_TEST_CASES: yaml_data}, schema_content=e2e_schema
            )
            return True
        except (YamlValidationException, ScannerError) as exc:
            structlogger.debug("e2e_test_generator.yaml_string_invalid", exc=exc)
            return False

    @staticmethod
    def remove_markdown_code_syntax(markdown_string: str) -> str:
        """Remove Markdown code formatting from the string.

        Args:
            markdown_string (str): string to be parsed.

        Returns:
            str: Parsed string.

        Example:
            >>> markdown_string = "```yaml\nkey: value\n```"
            >>> remove_markdown_formatting(markdown_string)
            'key: value'
        """
        if not markdown_string:
            return ""

        dedented_string = dedent(markdown_string)
        regex = r"^```.*\n|```$"
        return re.sub(regex, "", dedented_string, flags=re.MULTILINE).strip()

    @staticmethod
    async def generate_llm_response(prompt: str) -> Optional[str]:
        """Use LLM to generate a response.

        Args:
            prompt (str): the prompt to send to the LLM.

        Returns:
            Optional[str]: Generated response.
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
    ) -> Conversations:
        """Splits the data into conversations using empty row as a separator.

        Arguments:
            input_data (List[Dict]): The list of rows of the input file.

        Returns:
            Conversations: List of conversations
        """
        conversations = Conversations()
        conversation = Conversation()

        # Iterate through each row until the empty row separator is hit
        for row in input_data:
            # Remove empty values from the row
            row = {key: value for key, value in row.items() if value}

            if row:
                conversation_entry = ConversationEntry.from_dict(row)
                conversation.add_entry(conversation_entry)
            else:
                # If the current conversation exists, add it to the list
                if conversation:
                    conversations.add_conversation(conversation)
                    conversation = Conversation()

        # Add last conversation to the list as it might not have a separator after it
        if conversation:
            conversations.add_conversation(conversation)

        return conversations

    def get_and_validate_input_file_extension(self) -> str:
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

    def render_template(self, conversation: Conversation) -> str:
        """Renders a jinja2 template.

        Arguments:
            conversation (Conversation): The list of rows of a conversation.

        Returns:
            str: The rendered template string.
        """
        kwargs = {"conversation": conversation.as_list()}
        return Template(self.prompt_template).render(**kwargs)

    async def convert_single_conversation_into_test(
        self, conversation: Conversation
    ) -> str:
        """Uses LLM to convert a conversation to a YAML test case.

        Arguments:
            conversation (Conversation): The list of rows of a conversation.

        Returns:
            str: YAML representation of the conversation.
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
        self, conversations: Conversations
    ) -> str:
        """Generates test cases from the parsed data.

        Arguments:
            conversations (Conversations): The list of conversation rows.

        Returns:
            str: YAML representation of the test cases.
        """
        # Convert all conversations into YAML test cases asynchronously.
        tasks = [
            asyncio.ensure_future(
                self.convert_single_conversation_into_test(conversation)
            )
            for conversation in conversations.conversations
        ]
        results = await asyncio.gather(*tasks)

        structlogger.debug("e2e_test_generator.test_generation_finished")
        return "\n".join(results)

    def run(self) -> str:
        """Executes the E2E test conversion process: reads the file,
        splits the data into conversations, and generates test cases.
        """
        input_data = self.read_file()
        conversations = self.split_data_into_conversations(input_data)
        return asyncio.run(self.convert_conversations_into_tests(conversations))
