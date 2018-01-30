# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.training_data.message import Message
from rasa_nlu.training_data.training_data import TrainingData
from rasa_nlu.training_data.markdown_writer import MarkdownWriter
from rasa_nlu.training_data.markdown_reader import MarkdownReader


def transform_entity_synonyms(synonyms, known_synonyms=None):
    """Transforms the entity synonyms into a text->value dictionary"""
    entity_synonyms = known_synonyms if known_synonyms else {}
    for s in synonyms:
        if "value" in s and "synonyms" in s:
            for synonym in s["synonyms"]:
                entity_synonyms[synonym] = s["value"]
    return entity_synonyms