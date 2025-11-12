from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rasa.core.channels import UserMessage
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.importers.importer import TrainingDataImporter

if TYPE_CHECKING:
    from rasa.core.channels.channel import OutputChannel

PLACEHOLDER_IMPORTER = "__importer__"
PLACEHOLDER_MESSAGE = "__message__"
PLACEHOLDER_TRACKER = "__tracker__"
PLACEHOLDER_ENDPOINTS = "__endpoints__"
PLACEHOLDER_OUTPUT_CHANNEL = "__output_channel__"
RESERVED_PLACEHOLDERS: Dict[str, Any] = {
    PLACEHOLDER_IMPORTER: TrainingDataImporter,
    PLACEHOLDER_MESSAGE: List[UserMessage],
    PLACEHOLDER_TRACKER: DialogueStateTracker,
    PLACEHOLDER_ENDPOINTS: Optional[AvailableEndpoints],
    PLACEHOLDER_OUTPUT_CHANNEL: Optional["OutputChannel"],
}
