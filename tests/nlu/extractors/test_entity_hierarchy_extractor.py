from rasa.shared.nlu.constants import ENTITIES

from rasa.nlu.extractors.entity_hierarchy_extractor import EntityHierarchyExtractor
from rasa.nlu.extractors.entity_hierarchy_extractor import _topdownparser
from ruamel.yaml import YAML
from io import StringIO
from rasa.shared.nlu.training_data.message import Message

yaml = YAML(pure=False, typ="safe")

entities_yaml = """
festnetz:
  mappings:
  - value: true
    examples:
      - text: festnetz
      - text: anschluss
      - text: telefonanschluss
      - text: festnetzanschluss
      - text: leitung
      - text: ip-anschluss
      - ref: internet

internet:
  mappings:
  - value: true
    examples:
      - text: internet
      - ref: dsl
      - text: internetstörung
        alternatives:
        - internetstörungen
dsl:
  mappings:
  # no fixed value example!
  - examples:
      - text: dsl
      - composite: "{dsl_types}dsl"
      - text: vdsl2

dsl_types:
  config:
    structure_only: true
  mappings:
  - examples:
    - text: "a"
    - text: "v"
"""

entities_topdown = {
    "entities": {
        "festnetz": {"festnetz": True},
        "anschluss": {"festnetz": True},
        "telefonanschluss": {"festnetz": True},
        "festnetzanschluss": {"festnetz": True},
        "leitung": {"festnetz": True},
        "ip-anschluss": {"festnetz": True},
        "internet": {"festnetz": True, "internet": True},
        "dsl": {"festnetz": True, "internet": True, "dsl": "dsl"},
        "vdsl": {"festnetz": True, "internet": True, "dsl": "vdsl"},
        "vdsl2": {"festnetz": True, "internet": True, "dsl": "vdsl2"},
        "adsl": {"festnetz": True, "internet": True, "dsl": "adsl"},
        "internetstörung": {"festnetz": True, "internet": True},
    },
    "alternatives": {"internetstörungen": "internetstörung"},
}


def test_parser():
    stream = StringIO(entities_yaml)
    entities_definitions = yaml.load(stream=stream)
    parser_results = _topdownparser(data=entities_definitions)

    assert parser_results == entities_topdown, "Parser Result unexpected"


def test_mapping():
    EH = EntityHierarchyExtractor(
        component_config={
            "entityfile": None,
            "case_sensitive": False,
            "non_word_boundaries": "_öäüÖÄÜß-",
            "include_repeated_entities": False,
        },
        entityhierarchy=entities_topdown,
    )

    msg = Message.build(text="wir haben beim DSL internetstörungen")

    EH.process(message=msg)

    entities_expected = [
        {
            "entity": "festnetz",
            "start": 15,
            "end": 18,
            "value": True,
            "confidence": 1.0,
            "extractor": "EntityHierarchyExtractor",
        },
        {
            "entity": "internet",
            "start": 15,
            "end": 18,
            "value": True,
            "confidence": 1.0,
            "extractor": "EntityHierarchyExtractor",
        },
        {
            "entity": "dsl",
            "start": 15,
            "end": 18,
            "value": "dsl",
            "confidence": 1.0,
            "extractor": "EntityHierarchyExtractor",
        },
    ]
    print(msg.get(ENTITIES))
    assert msg.get(ENTITIES) == entities_expected, "Entity extraction failed"
