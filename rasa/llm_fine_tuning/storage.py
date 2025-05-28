import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import rasa.shared.utils.io
import rasa.utils.io
from rasa.llm_fine_tuning.conversations import Conversation
from rasa.shared.utils.yaml import write_yaml

if TYPE_CHECKING:
    from rasa.e2e_test.e2e_test_case import TestSuite
    from rasa.llm_fine_tuning.llm_data_preparation_module import LLMDataExample
    from rasa.llm_fine_tuning.train_test_split_module import DataExampleFormat


class StorageType(Enum):
    FILE = "file"
    # We might want to add other storage systems in the future
    # CLOUD = "cloud"
    # DATABASE = "database"


class StorageStrategy(ABC):
    @abstractmethod
    def write_conversation(
        self, conversation: Conversation, storage_location: Optional[str] = None
    ) -> None:
        pass

    def write_llm_data(
        self, llm_data: List["LLMDataExample"], storage_location: Optional[str]
    ) -> None:
        pass

    def write_formatted_finetuning_data(
        self,
        formatted_data: List["DataExampleFormat"],
        module_storage_location: Optional[str],
        file_name: Optional[str],
    ) -> None:
        pass

    def write_e2e_test_suite_to_yaml_file(
        self,
        e2e_test_suite: "TestSuite",
        module_storage_location: Optional[str],
        file_name: Optional[str],
    ) -> None:
        pass


class FileStorageStrategy(StorageStrategy):
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def write_conversation(
        self, conversation: Conversation, storage_location: Optional[str] = None
    ) -> None:
        file_name = os.path.basename(
            conversation.original_e2e_test_case.file or f"{conversation.name}.yaml"
        )
        file_path = self._get_file_path(storage_location, file_name)
        self._create_output_dir(file_path)

        rasa.utils.io.write_yaml(conversation.as_dict(), file_path)

    def _get_file_path(
        self, storage_location: Optional[str], file_name: Optional[str] = None
    ) -> Path:
        if storage_location:
            output = f"{self.output_dir}/{storage_location}"
        else:
            output = self.output_dir

        if file_name:
            return Path(f"{output}/{file_name}")
        return Path(output)

    @staticmethod
    def _create_output_dir(file_path: Path) -> None:
        if (
            str(file_path).endswith(".jsonl")
            or str(file_path).endswith(".yaml")
            or str(file_path).endswith(".yml")
        ):
            file_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            file_path.mkdir(exist_ok=True, parents=True)

    def write_llm_data(
        self, llm_data: List["LLMDataExample"], storage_location: Optional[str]
    ) -> None:
        file_path = self._get_file_path(storage_location)
        self._create_output_dir(file_path)

        with open(str(file_path), "w", encoding="utf-8") as outfile:
            for example in llm_data:
                json.dump(example.as_dict(), outfile, ensure_ascii=False)
                outfile.write("\n")

    def write_formatted_finetuning_data(
        self,
        formatted_data: List["DataExampleFormat"],
        module_storage_location: Optional[str],
        file_name: Optional[str],
    ) -> None:
        file_path = self._get_file_path(module_storage_location, file_name)
        self._create_output_dir(file_path)

        with open(str(file_path), "w", encoding="utf-8") as file:
            for example in formatted_data:
                json.dump(example.as_dict(), file, ensure_ascii=False)
                file.write("\n")

    def write_e2e_test_suite_to_yaml_file(
        self,
        e2e_test_suite: "TestSuite",
        module_storage_location: Optional[str],
        file_name: Optional[str],
    ) -> None:
        """Write the e2e TestSuite data to a YAML file.

        Args:
            e2e_test_suite: The TestSuite object to write.
            module_storage_location: The location where the file should be stored.
            file_name: The path to the file where the data should be written.
        """
        file_path = self._get_file_path(module_storage_location, file_name)
        self._create_output_dir(file_path)
        write_yaml(e2e_test_suite.as_dict(), str(file_path))


class StorageContext:
    def __init__(self, strategy: StorageStrategy) -> None:
        self.strategy = strategy

    def write_conversation(
        self, conversation: Conversation, sub_dir: Optional[str] = None
    ) -> None:
        self.strategy.write_conversation(conversation, sub_dir)

    def write_conversations(
        self, conversations: List[Conversation], sub_dir: Optional[str] = None
    ) -> None:
        for conversation in conversations:
            self.write_conversation(conversation, sub_dir)

    def write_llm_data(
        self, llm_data: List["LLMDataExample"], file_path: Optional[str] = None
    ) -> None:
        self.strategy.write_llm_data(llm_data, file_path)

    def write_formatted_finetuning_data(
        self,
        formatted_data: List["DataExampleFormat"],
        module_storage_location: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> None:
        self.strategy.write_formatted_finetuning_data(
            formatted_data, module_storage_location, file_name
        )

    def write_e2e_test_suite_to_yaml_file(
        self,
        e2e_test_suite: "TestSuite",
        module_storage_location: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> None:
        self.strategy.write_e2e_test_suite_to_yaml_file(
            e2e_test_suite, module_storage_location, file_name
        )
