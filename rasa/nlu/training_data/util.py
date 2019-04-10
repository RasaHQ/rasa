# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)


def transform_entity_synonyms(synonyms, known_synonyms=None):
    """Transforms the entity synonyms into a text->value dictionary"""
    entity_synonyms = known_synonyms if known_synonyms else {}
    for s in synonyms:
        if "value" in s and "synonyms" in s:
            for synonym in s["synonyms"]:
                entity_synonyms[synonym] = s["value"]
    return entity_synonyms


def check_duplicate_synonym(entity_synonyms, text, syn, context_str=""):
    if text in entity_synonyms and entity_synonyms[text] != syn:
        logger.warning("Found inconsistent entity synonyms while {0}, "
                       "overwriting {1}->{2} "
                       "with {1}->{3} during merge"
                       "".format(context_str, text, entity_synonyms[text], syn))
