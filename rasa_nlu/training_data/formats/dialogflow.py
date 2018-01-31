from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_nlu.training_data import Message, TrainingData
from rasa_nlu.training_data.formats import JsonTrainingDataReader
from rasa_nlu import utils

logger = logging.getLogger(__name__)


class DialogflowReader(JsonTrainingDataReader):
    def read_from_json(self, js, **kwargs):
        # type: ([Text]) -> TrainingData
        """Loads training data stored in the Dialogflow data format."""

        language = kwargs["language"]
        training_examples = []
        entity_synonyms = {}
        for filename in files:
            data = utils.read_json_file(filename)
            # Language specific extensions
            usersays_file_ext = '_usersays_{}.json'.format(language)
            synonyms_file_ext = '_entries_{}.json'.format(language)
            if filename.endswith(usersays_file_ext):
                synonyms_filename = filename.replace(usersays_file_ext, '.json')
                root_f_data = utils.read_json_file(synonyms_filename)
                intent = root_f_data.get("name")

                for s in data:
                    text = "".join([chunk["text"] for chunk in s.get("data")])
                    # add entities to each token, if available
                    entities = []
                    for e in [chunk
                              for chunk in s.get("data")
                              if "alias" in chunk or "meta" in chunk]:
                        start = text.find(e["text"])
                        end = start + len(e["text"])
                        val = text[start:end]
                        entity_type = e["alias"] if "alias" in e else e["meta"]
                        if entity_type != u'@sys.ignore':
                            entities.append(
                                    {
                                        "entity": entity_type,
                                        "value": val,
                                        "start": start,
                                        "end": end
                                    }
                            )
                    data = {}
                    if intent:
                        data["intent"] = intent
                    if entities is not None:
                        data["entities"] = entities
                    training_examples.append(Message(text, data))

            elif filename.endswith(synonyms_file_ext):
                # create synonyms dictionary
                for entry in data:
                    if "value" in entry and "synonyms" in entry:
                        for synonym in entry["synonyms"]:
                            entity_synonyms[synonym] = entry["value"]
        return TrainingData(training_examples, entity_synonyms)