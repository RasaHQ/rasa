from typing import List, Optional

from rasa.core.channels import UserMessage
from rasa.core.utils import AvailableEndpoints
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.importers.importer import TrainingDataImporter

PLACEHOLDER_IMPORTER = "__importer__"
PLACEHOLDER_MESSAGE = "__message__"
PLACEHOLDER_TRACKER = "__tracker__"
PLACEHOLDER_ENDPOINTS = "__endpoints__"
RESERVED_PLACEHOLDERS = {
    PLACEHOLDER_IMPORTER: TrainingDataImporter,
    PLACEHOLDER_MESSAGE: List[UserMessage],
    PLACEHOLDER_TRACKER: DialogueStateTracker,
    PLACEHOLDER_ENDPOINTS: Optional[AvailableEndpoints],
}
