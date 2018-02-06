from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import logging
import random
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from spacy.util import minibatch
from tqdm import tqdm

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc

logger = logging.getLogger(__name__)


class SpacyEntityExtractor(EntityExtractor):
    name = "ner_spacy"

    provides = ["entities"]

    requires = ["spacy_doc", "spacy_nlp"]

    def __init__(self):
        self.spacy_nlp = None

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> Dict[Text, Any]

        ner_config = config.get('ner_spacy')

        nlp = kwargs['spacy_nlp']

        # get the ner pipe
        ner = nlp.get_pipe('ner')

        training_ner_data = []
        if training_data.entity_examples:
            for example in training_data.entity_examples:
                entities = [(t['start'], t['end'], t['entity']) for t in example.get('entities')]
                for _, _, entity in entities:
                    ner.add_label(entity)

                training_ner_data.append((example.text, {'entities': entities}))

        self.__train_ner(ner_config, nlp, training_ner_data)

        ner.cfg['rasa_updated'] = True
        return {'spacy_nlp': nlp}

    def __train_ner(self, ner_config, nlp, training_ner_data):
        batch_size = ner_config.get('batch_size', 16)
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            logger.info('start training ner')
            optimizer = nlp.begin_training()
            epochs = ner_config.get('epochs')
            for it in range(epochs):
                random.shuffle(training_ner_data)
                losses = {}

                batches = minibatch(training_ner_data, size=batch_size)
                progress = tqdm(batches, total=len(training_ner_data) / batch_size, miniters=10)
                for batch in progress:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                               losses=losses)
                    progress.set_description_str('epoch %d/%d, loss: %s' % (it + 1, epochs, str(losses)))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.extract_entities(message.get("spacy_doc")))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def extract_entities(self, doc):
        # type: (Doc) -> List[Dict[Text, Any]]

        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents]
        return entities
