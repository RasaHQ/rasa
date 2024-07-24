import os
from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Optional

import rasa.shared.utils.io
import rasa.utils.io
from rasa.llm_fine_tuning.conversations import Conversation


class StorageType(Enum):
    FILE = "file"
    # We might want to add other storage systems in the future
    # CLOUD = "cloud"
    # DATABASE = "database"


class StorageStrategy(ABC):
    @abstractmethod
    def write(
        self, conversation: Conversation, storage_location: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def read(self) -> Conversation:
        pass


class FileStorageStrategy(StorageStrategy):
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def write(
        self, conversation: Conversation, storage_location: Optional[str] = None
    ) -> None:
        file_name = os.path.basename(
            conversation.original_e2e_test_case.file or f"{conversation.name}.yaml"
        )
        if storage_location:
            output = f"{self.output_dir}/{storage_location}"
        else:
            output = self.output_dir
        rasa.shared.utils.io.create_directory(output)

        rasa.utils.io.write_yaml(conversation.as_dict(), f"{output}/{file_name}")

    def read(self) -> Conversation:
        raise NotImplementedError()


class StorageContext:
    def __init__(self, strategy: StorageStrategy) -> None:
        self.strategy = strategy

    def write_conversation(
        self, conversation: Conversation, sub_dir: Optional[str] = None
    ) -> None:
        self.strategy.write(conversation, sub_dir)

    def write_conversations(
        self, conversations: List[Conversation], sub_dir: Optional[str] = None
    ) -> None:
        for conversation in conversations:
            self.write_conversation(conversation, sub_dir)

    def read_conversation(self) -> None:
        raise NotImplementedError()
