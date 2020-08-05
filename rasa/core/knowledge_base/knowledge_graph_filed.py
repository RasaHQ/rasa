# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Mainly borrowed from `allennlp.data.fields.knowledge_graph_filed.py`

############################################################
#                       NOTICE                             #
#   we maintain this file for not sorting the entities.    #
#   which is important for our model!                      #
############################################################

Author: Qian Liu
"""


"""
A ``KnowledgeGraph`` is a graphical representation of some structured knowledge source: say a
table, figure or an explicit knowledge base.
"""

from typing import Dict, List, Set


class KnowledgeGraph:
    """
    A ``KnowledgeGraph`` represents a collection of entities and their relationships.

    The ``KnowledgeGraph`` currently stores (untyped) neighborhood information and text
    representations of each entity (if there is any).

    The knowledge base itself can be a table (like in WikitableQuestions), a figure (like in NLVR)
    or some other structured knowledge source. This abstract class needs to be inherited for
    implementing the functionality appropriate for a given KB.

    All of the parameters listed below are stored as public attributes.

    Parameters
    ----------
    entities : ``Set[str]``
        The string identifiers of the entities in this knowledge graph.  We sort this set and store
        it as a list.  The sorting is so that we get a guaranteed consistent ordering across
        separate runs of the code.
    neighbors : ``Dict[str, List[str]]``
        A mapping from string identifiers to other string identifiers, denoting which entities are
        neighbors in the graph.
    entity_text : ``Dict[str, str]``
        If you have additional text associated with each entity (other than its string identifier),
        you can store that here.  This might be, e.g., the text in a table cell, or the description
        of a wikipedia entity.
    """

    def __init__(
        self,
        entities: List[str],
        neighbors: Dict[str, List[str]],
        neighbors_with_table: Dict[str, List[str]],
        entity_text: Dict[str, str] = None,
    ) -> None:
        self.entities = entities
        self.neighbors = neighbors
        self.neighbors_with_table = neighbors_with_table
        self.entity_text = entity_text

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented
