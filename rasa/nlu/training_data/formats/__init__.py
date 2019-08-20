from rasa.nlu.training_data.formats.dialogflow import DialogFlowReader
from rasa.nlu.training_data.formats.luis import LuisReader
from rasa.nlu.training_data.formats.markdown import MarkdownReader, MarkdownWriter
from rasa.nlu.training_data.formats.rasa import RasaReader, RasaWriter
from rasa.nlu.training_data.formats.wit import WitReader

from rasa.nlu.training_data.formats.dialogflow import (
    DIALOGFLOW_ENTITIES,
    DIALOGFLOW_INTENT,
)


class SupportedFormats:
    WIT = "wit"
    LUIS = "luis"
    RASA = "rasa_nlu"
    MARKDOWN = "md"
    UNK = "unk"
    DIALOGFLOW_RELEVANT = {DIALOGFLOW_ENTITIES, DIALOGFLOW_INTENT}
