import atexit
from datetime import date, datetime
from collections import defaultdict
import json
import os
import re
import signal
import threading


class DialogueFileLogger:

    conversations = dict()
    lock = threading.Lock()
    counter = 1

    @staticmethod
    def add_user_statement(user_id: str, text: str, intent: str, entities: str):
        if str(user_id) not in DialogueFileLogger.conversations:
            DialogueFileLogger.conversations[str(user_id)] =\
                    SingleDialogueFileLogger(DialogueFileLogger._generate_file_name())
        DialogueFileLogger.conversations[str(user_id)].add_user_statement(text, intent, entities)

    @staticmethod
    def add_bot_statements(user_id: str, utterances: list, action: str):
        if str(user_id) not in DialogueFileLogger.conversations:
            DialogueFileLogger.conversations[str(user_id)] =\
                    SingleDialogueFileLogger(DialogueFileLogger._generate_file_name())
        for utterance in utterances:
            DialogueFileLogger.conversations[str(user_id)].add_bot_statement(utterance, action)

    @staticmethod
    def _generate_file_name():
        with DialogueFileLogger.lock:
            name = str(datetime.now()) + '-' + str(DialogueFileLogger.counter)
            DialogueFileLogger.counter = (DialogueFileLogger.counter + 1) % 10000
            return name

    @staticmethod
    def save_to_file(user_id: str):
        DialogueFileLogger.conversations[str(user_id)].save_to_file()
        del DialogueFileLogger.conversations[str(user_id)]

    @staticmethod
    def save_all():
        user_ids = [user_id for user_id in DialogueFileLogger.conversations]
        for user_id in user_ids:
            DialogueFileLogger.save_to_file(user_id)


atexit.register(DialogueFileLogger.save_all)
signal.signal(signal.SIGTERM, DialogueFileLogger.save_all)
signal.signal(signal.SIGINT, DialogueFileLogger.save_all)


class SingleDialogueFileLogger:
    def __init__(self, filename):
        self.filename = filename
        self.conversation = dict()
        self.conversation["version"] = '1'
        self.conversation["date"] = str(date.today())
        self.conversation["statements"] = []

    def add_user_statement(self, text: str, intent: str, entities: str):
        statement = dict()
        statement["speaker"] = "User"
        statement["time"] = self._get_time()
        statement["text"] = str(text)
        statement["intent"] = json.loads(str(intent).replace('\'', '\"'))
        statement["entities"] = str(entities)
        self.conversation["statements"].append(statement)
    
    def add_bot_statement(self, utterance: str, action: str):
        text = re.search(r'BotUttered\(text\: (.*?)\,', str(utterance))
        if text is not None:  # Ignore empty bot messages
            statement = dict()
            statement["speaker"] = "Bot"
            statement["time"] = self._get_time()
            statement["action"] = str(action)
            statement["text"] = text.group(1)
            self.conversation["statements"].append(statement)

    def _get_time(self) -> str:
        now = datetime.now()
        return str(now.strftime("%H:%M:%S"))

    def save_to_file(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open("logs/" + self.filename + ".json", "w") as file:
            json.dump(self.conversation, file, indent=4)
